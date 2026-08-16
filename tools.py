"""Core execution logic for the ``ml_paper_analyst`` skill.

Module layout (all classes are async and share one :class:`HTTPClient` when
driven through :class:`skill.MLPaperAnalystSkill`):

======================  ====================================================
Component               Purpose
======================  ====================================================
:class:`AsyncRateLimiter`   Per-host pacing (arXiv 1 req/3s, S2 ~1 rps).
HTTP helpers            Exponential-backoff retry on 429/5xx/timeouts.
:class:`DiskCache`      Best-effort JSON disk cache with TTL.
:class:`OpenAIChatBackend` Default LLM backend (OpenAI-compatible API).
:class:`Searcher`       Cross-source search + query expansion + filters.
:class:`Extractor`      ar5iv / arXiv-HTML parsing with PDF fallback.
:class:`Analyzer`       Four-aspect LLM analysis with heuristic fallback.
:class:`Synthesizer`    Comparison matrix, BibTeX, literature review.
======================  ====================================================

The four module-level functions at the bottom mirror the Agent tool schema
declared in ``manifest.json``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from urllib.parse import quote_plus, urlsplit

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError

from .config import DEFAULT_CONFIG, SUPPORTED_SOURCES, MLPaperAnalystConfig
from .prompts import (
    ACADEMIC_ANALYST_SYSTEM_PROMPT,
    CONTRIBUTION_USER_TEMPLATE,
    CRITIQUE_USER_TEMPLATE,
    EXPERIMENTS_USER_TEMPLATE,
    METHODOLOGY_USER_TEMPLATE,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    QUERY_EXPANSION_USER_TEMPLATE,
    SINGLE_PASS_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
    render_analysis_context,
    render_synthesis_digest,
)
from .schemas import (
    AlgorithmInfo,
    AnalysisReport,
    CoreContribution,
    Critique,
    ExperimentalValidation,
    FigureInfo,
    FormulaInfo,
    LiteratureReview,
    MethodologyBreakdown,
    PaperContent,
    PaperMetadata,
    SearchQuery,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncRateLimiter",
    "DiskCache",
    "HTTPClient",
    "LLMBackend",
    "OpenAIChatBackend",
    "Searcher",
    "Extractor",
    "Analyzer",
    "Synthesizer",
    "search_ml_papers",
    "fetch_and_parse_paper",
    "analyze_single_paper",
    "synthesize_literature_review",
]

T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Small text utilities                                                        #
# --------------------------------------------------------------------------- #


def _clean_ws(text: str) -> str:
    """Collapse all whitespace runs into single spaces.

    Args:
        text: Arbitrary text.

    Returns:
        Cleaned single-line-ish text.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def _slugify(text: str, max_len: int = 48) -> str:
    """Convert a section title to a lowercase snake_case slug.

    Args:
        text: Input title.
        max_len: Maximum slug length.

    Returns:
        Slug containing only ``[a-z0-9_]``.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:max_len].rstrip("_") or "unnamed"


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object embedded in an LLM response.

    Handles markdown code fences and prose around the JSON payload.

    Args:
        text: Raw LLM output.

    Returns:
        Parsed dict, or ``None`` when no parsable object exists.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- #
# HTTP infrastructure: rate limiting + exponential backoff                    #
# --------------------------------------------------------------------------- #


class AsyncRateLimiter:
    """Enforce a minimum interval between requests (per host).

    Args:
        min_interval: Minimum seconds between consecutive acquisitions.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        """Block until the configured interval has elapsed since last call."""
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


#: HTTP statuses on which we retry with exponential backoff.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504, 520, 529}


class HTTPClient:
    """Async HTTP client with per-host rate limiting, bounded concurrency and
    exponential-backoff retries (honouring ``Retry-After`` when present).

    Args:
        config: Skill configuration.
    """

    def __init__(self, config: MLPaperAnalystConfig) -> None:
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._limiters: Dict[str, AsyncRateLimiter] = {}

    async def __aenter__(self) -> "HTTPClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying aiohttp session if it was created."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create the aiohttp session inside a running loop."""
        if self._session is None or self._session.closed:
            headers = {"User-Agent": self._config.user_agent}
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _limiter(self, host: str) -> AsyncRateLimiter:
        """Return (creating if needed) the rate limiter for ``host``."""
        if host not in self._limiters:
            interval = self._config.rate_limits.get(
                host, self._config.default_rate_limit_interval
            )
            self._limiters[host] = AsyncRateLimiter(interval)
        return self._limiters[host]

    def _backoff_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        """Compute the next retry delay.

        Args:
            attempt: Zero-based attempt number.
            retry_after: Raw ``Retry-After`` header value, when present.

        Returns:
            Delay in seconds (jittered, capped).
        """
        cfg = self._config
        delay = min(
            cfg.backoff_max_delay, cfg.backoff_base * (2 ** attempt)
        ) * (1.0 + random.uniform(0.0, cfg.backoff_jitter))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return min(delay, cfg.backoff_max_delay + 30.0)

    async def _run(
        self,
        host: str,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute ``factory`` with semaphore, rate limiting and retries.

        Args:
            host: Host name used for rate limiting.
            factory: Zero-arg coroutine producing the response value; it must
                raise :class:`aiohttp.ClientResponseError` on HTTP errors.

        Returns:
            Whatever ``factory`` returns.

        Raises:
            aiohttp.ClientError: Propagated after retries are exhausted.
            asyncio.TimeoutError: Propagated after retries are exhausted.
        """
        limiter = self._limiter(host)
        max_attempts = self._config.max_retries + 1
        last_error: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                async with self._semaphore:
                    await limiter.acquire()
                    return await factory()
            except aiohttp.ClientResponseError as exc:
                last_error = exc
                if exc.status not in _RETRYABLE_STATUSES:
                    raise
                if attempt >= max_attempts - 1:
                    raise
                retry_after = None
                if exc.headers is not None:
                    retry_after = exc.headers.get("Retry-After")
                delay = self._backoff_delay(attempt, retry_after)
                logger.debug(
                    "HTTP %s from %s — retrying in %.1fs (attempt %d/%d)",
                    exc.status, host, delay, attempt + 1, max_attempts - 1,
                )
                await asyncio.sleep(delay)
            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    raise
                delay = self._backoff_delay(attempt, None)
                logger.debug(
                    "Timeout from %s — retrying in %.1fs", host, delay
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    def _headers_for(self, url: str) -> Dict[str, str]:
        """Build extra headers for a URL (S2 API key injection)."""
        headers: Dict[str, str] = {}
        host = urlsplit(url).netloc
        if (
            host == "api.semanticscholar.org"
            and self._config.semantic_scholar_api_key
        ):
            headers["x-api-key"] = self._config.semantic_scholar_api_key
        return headers

    async def get_text(self, url: str, **request_kwargs: Any) -> str:
        """GET a URL and return the response body as text.

        Args:
            url: Absolute URL.
            **request_kwargs: Extra kwargs forwarded to
                :meth:`aiohttp.ClientSession.request`.

        Returns:
            Decoded response text.

        Raises:
            aiohttp.ClientError: On HTTP/network failure after retries.
        """
        host = urlsplit(url).netloc

        async def factory() -> str:
            session = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
            async with session.get(
                url,
                raise_for_status=True,
                timeout=timeout,
                headers=self._headers_for(url),
                **request_kwargs,
            ) as resp:
                return await resp.text(errors="replace")

        return await self._run(host, factory)

    async def get_json(self, url: str, **request_kwargs: Any) -> Any:
        """GET a URL and parse the response as JSON.

        Args:
            url: Absolute URL.
            **request_kwargs: Extra kwargs forwarded to the request call.

        Returns:
            Parsed JSON value.

        Raises:
            aiohttp.ClientError: On HTTP/network failure after retries.
        """
        host = urlsplit(url).netloc

        async def factory() -> Any:
            session = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
            async with session.get(
                url,
                raise_for_status=True,
                timeout=timeout,
                headers=self._headers_for(url),
                **request_kwargs,
            ) as resp:
                return await resp.json(content_type=None)

        return await self._run(host, factory)

    async def post_json(self, url: str, json_body: Any) -> Any:
        """POST a JSON body and parse the JSON response.

        Args:
            url: Absolute URL.
            json_body: JSON-serialisable request body.

        Returns:
            Parsed JSON value.

        Raises:
            aiohttp.ClientError: On HTTP/network failure after retries.
        """
        host = urlsplit(url).netloc

        async def factory() -> Any:
            session = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
            async with session.post(
                url,
                json=json_body,
                raise_for_status=True,
                timeout=timeout,
                headers=self._headers_for(url),
            ) as resp:
                return await resp.json(content_type=None)

        return await self._run(host, factory)

    async def get_bytes(self, url: str, max_bytes: int = 40_000_000) -> bytes:
        """GET binary content (e.g. a PDF) with a size cap.

        Args:
            url: Absolute URL.
            max_bytes: Hard cap on downloaded bytes.

        Returns:
            Response payload.

        Raises:
            aiohttp.ClientError: On HTTP/network failure after retries.
            ValueError: When the payload exceeds ``max_bytes``.
        """
        host = urlsplit(url).netloc

        async def factory() -> bytes:
            session = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
            async with session.get(
                url,
                raise_for_status=True,
                timeout=timeout,
                headers=self._headers_for(url),
            ) as resp:
                payload = await resp.read()
                if len(payload) > max_bytes:
                    raise ValueError(
                        f"payload from {url} exceeds {max_bytes} bytes"
                    )
                return payload

        return await self._run(host, factory)


# --------------------------------------------------------------------------- #
# Disk cache                                                                  #
# --------------------------------------------------------------------------- #


class DiskCache:
    """Best-effort JSON disk cache with TTL.

    Failures (permissions, corrupt files, disk full) never raise: the cache
    degrades to a no-op so the main pipeline is never blocked.

    Args:
        config: Skill configuration providing the cache directory and TTL.
    """

    def __init__(self, config: MLPaperAnalystConfig) -> None:
        self._config = config
        self._dir: Optional[Path] = None
        if config.cache_enabled:
            try:
                self._dir = config.cache_dir.expanduser()
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Cache disabled (cannot create dir): %s", exc)
                self._dir = None

    @property
    def enabled(self) -> bool:
        """Whether the cache is active."""
        return self._dir is not None

    @staticmethod
    def _key_hash(key: str) -> str:
        """Hash a cache key into a filesystem-safe filename."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Fetch a cached value.

        Args:
            key: Logical cache key.

        Returns:
            Cached payload, or ``None`` on miss/expiry/error.
        """
        if self._dir is None:
            return None
        path = self._dir / f"{self._key_hash(key)}.json"
        try:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as fh:
                wrapper = json.load(fh)
            if time.time() - wrapper.get("ts", 0) > wrapper.get("ttl", 0):
                path.unlink(missing_ok=True)
                return None
            return wrapper.get("payload")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.debug("Cache read failed for %s: %s", key, exc)
            return None

    def set(self, key: str, payload: Any, ttl: Optional[int] = None) -> None:
        """Store a JSON-serialisable payload.

        Args:
            key: Logical cache key.
            payload: JSON-serialisable value.
            ttl: Optional TTL override in seconds.
        """
        if self._dir is None:
            return
        path = self._dir / f"{self._key_hash(key)}.json"
        wrapper = {
            "ts": time.time(),
            "ttl": ttl if ttl is not None else self._config.cache_ttl,
            "payload": payload,
        }
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(wrapper, fh, ensure_ascii=False)
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Cache write failed for %s: %s", key, exc)
            tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# LLM backend abstraction                                                     #
# --------------------------------------------------------------------------- #


class LLMBackend:
    """Minimal protocol every LLM backend must satisfy.

    The hosting Agent can inject its own model by implementing
    ``complete``; the skill is model-agnostic.
    """

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> str:
        """Generate a completion for a system+user prompt pair.

        Args:
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum generated tokens.

        Returns:
            The completion text.

        Raises:
            RuntimeError: If called on the protocol base class.
        """
        raise RuntimeError("LLMBackend is a protocol; implement complete()")


class OpenAIChatBackend:
    """Default LLM backend talking to any OpenAI-compatible chat API.

    Works with OpenAI, Azure-compatible gateways, vLLM, Ollama's OpenAI
    endpoint, etc. — anything accepting ``/v1/chat/completions``.

    Args:
        model: Model identifier (e.g. ``gpt-4o-mini``).
        api_key: API key; falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional custom base URL (self-hosted gateways).
        json_mode: When True, requests JSON response format.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        json_mode: bool = False,
    ) -> None:
        try:
            import openai  # Imported lazily so the skill works without it.
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "The 'openai' package is required for OpenAIChatBackend. "
                "Install it via `pip install openai` or inject a custom "
                "LLMBackend implementation."
            ) from exc
        self.model = model
        self._json_mode = json_mode
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> str:
        """Generate a chat completion.

        Args:
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum generated tokens.

        Returns:
            Assistant message text.

        Raises:
            RuntimeError: When the API call fails after retries.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - provider SDK mixes types
                last_error = exc
                await asyncio.sleep(min(8.0, 1.5 * (2 ** attempt)))
        raise RuntimeError(f"LLM call failed after retries: {last_error}")


def resolve_llm_backend(
    config: MLPaperAnalystConfig,
    explicit: Optional[LLMBackend],
) -> Optional[LLMBackend]:
    """Decide which LLM backend to use.

    Args:
        config: Skill configuration.
        explicit: Backend explicitly injected by the caller (highest priority).

    Returns:
        A backend instance, or ``None`` when no LLM is available (the skill
        then degrades to heuristic analysis).
    """
    if explicit is not None:
        return explicit
    if config.openai_api_key:
        return OpenAIChatBackend(
            model=config.llm_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            json_mode=config.llm_json_mode,
        )
    return None


# --------------------------------------------------------------------------- #
# Venue utilities                                                             #
# --------------------------------------------------------------------------- #

#: Canonical top venue -> lowercase alias set.
_VENUE_ALIASES: Dict[str, Set[str]] = {
    "NeurIPS": {"nips", "neurips", "neural information processing systems",
                "advances in neural information processing systems"},
    "ICML": {"icml", "international conference on machine learning"},
    "ICLR": {"iclr", "international conference on learning representations"},
    "CVPR": {"cvpr", "conference on computer vision and pattern recognition"},
    "ICCV": {"iccv", "international conference on computer vision"},
    "ECCV": {"eccv", "european conference on computer vision"},
    "ACL": {"acl", "annual meeting of the association for computational linguistics"},
    "EMNLP": {"emnlp", "empirical methods in natural language processing"},
    "NAACL": {"naacl", "north american chapter of the acl"},
    "KDD": {"kdd", "knowledge discovery and data mining"},
    "WWW": {"www", "the web conference", "world wide web"},
    "SIGIR": {"sigir", "research and development in information retrieval"},
    "AAAI": {"aaai", "association for the advancement of artificial intelligence"},
    "IJCAI": {"ijcai", "international joint conference on artificial intelligence"},
    "SIGGRAPH": {"siggraph"},
    "SIGGRAPH Asia": {"siggraph asia"},
    "ICRA": {"icra", "international conference on robotics and automation",
             "robotics and automation"},
    "IROS": {"iros", "intelligent robots and systems"},
    "Nature": {"nature"},
    "Science": {"science"},
    "Nature Machine Intelligence": {"nature machine intelligence"},
    "Nature Communications": {"nature communications"},
    "TPAMI": {"tpami", "pattern analysis and machine intelligence"},
    "JMLR": {"jmlr", "journal of machine learning research"},
    "TMLR": {"tmlr", "transactions on machine learning research"},
}

#: Venues treated as journals when synthesising BibTeX.
_JOURNAL_VENUES: Set[str] = {
    "nature", "science", "nature machine intelligence",
    "nature communications", "tpami", "jmlr", "tmlr",
}

TOP_VENUES: List[str] = sorted(_VENUE_ALIASES.keys())

_COMMENT_VENUE_RE = re.compile(
    r"(?:accepted(?:\s+to)?|published\s+in|to\s+appear(?:\s+in|\s+at)?)\s+"
    r"(?:at\s+|in\s+|by\s+)?"
    r"([A-Za-z][A-Za-z0-9&\.\- ]{2,60}?)"
    r"(?=\s*[(,;.]|\s+20\d{2}|$)",
    re.IGNORECASE,
)


def _norm_venue(name: str) -> str:
    """Normalise a venue string for tolerant comparison."""
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _venue_from_comment(comment: Optional[str]) -> Optional[str]:
    """Try to recover a venue name from the free-form arXiv comment field.

    Args:
        comment: Raw arXiv comment (e.g. ``Accepted to CVPR 2024``).

    Returns:
        Extracted venue name or ``None``.
    """
    if not comment:
        return None
    match = _COMMENT_VENUE_RE.search(comment)
    if match:
        return match.group(1).strip()
    return None


def _expand_venue_filter(venue_filter: List[str]) -> Set[str]:
    """Expand user-supplied venue names into a normalised alias set.

    Handles canonical names ("CVPR"), canonical aliases ("NIPS") and full
    conference names ("Conference on Computer Vision and Pattern
    Recognition") by looking the alias up in both directions.

    Args:
        venue_filter: User-requested venue names (any casing/alias).

    Returns:
        Set of normalised strings, any of which counts as a match.
    """
    wanted: Set[str] = set()
    for name in venue_filter:
        norm = _norm_venue(name)
        if not norm:
            continue
        wanted.add(norm)
        for alias in _VENUE_ALIASES.get(name, set()):
            wanted.add(_norm_venue(alias))
        # Reverse lookup: the user may have supplied an alias or the full
        # conference name of a canonical venue.
        for canonical, aliases in _VENUE_ALIASES.items():
            canonical_norm = _norm_venue(canonical)
            alias_norms = {_norm_venue(a) for a in aliases}
            if norm == canonical_norm or norm in alias_norms:
                wanted.add(canonical_norm)
                wanted.update(alias_norms)
    wanted.discard("")
    return wanted


def venue_matches(paper: PaperMetadata, venue_filter: List[str]) -> bool:
    """Check whether a paper's venue matches any of the requested venues.

    Matching tolerates aliases, substrings and venues recovered from the
    arXiv comment field.

    Args:
        paper: Paper metadata.
        venue_filter: User-requested venue names (any casing/alias).

    Returns:
        ``True`` when the paper plausibly appeared in one of the venues.
    """
    if not venue_filter:
        return True
    wanted = _expand_venue_filter(venue_filter)

    candidates: List[str] = []
    if paper.venue:
        candidates.append(paper.venue)
    extracted = _venue_from_comment(paper.comment)
    if extracted:
        candidates.append(extracted)
    if paper.comment:
        candidates.append(paper.comment)

    for raw in candidates:
        norm = _norm_venue(raw)
        if not norm:
            continue
        for target in wanted:
            if not target:
                continue
            if norm == target or target in norm or norm in target:
                return True
    return False


# --------------------------------------------------------------------------- #
# Module 1 — Searcher                                                         #
# --------------------------------------------------------------------------- #

#: Curated cross-expansion taxonomy: research sub-field trigger terms ->
#: canonical related keywords. Used when no LLM is configured.
QUERY_EXPANSION_TAXONOMY: Dict[str, List[str]] = {
    "text to 3d": [
        "text-to-3d generation", "score distillation sampling", "SDS loss",
        "dreamfusion", "nerf", "neural radiance fields", "gaussian splatting",
        "3d gaussian splatting", "differentiable rendering", "latent 3d diffusion",
    ],
    "diffusion": [
        "denoising diffusion probabilistic models", "ddpm", "score matching",
        "latent diffusion", "stable diffusion", "flow matching",
        "consistency models", "classifier-free guidance", "diffusion transformer",
    ],
    "large language model": [
        "instruction tuning", "rlhf", "chain-of-thought", "in-context learning",
        "lora", "mixture of experts", "long context", "speculative decoding",
        "retrieval-augmented generation", "scaling laws",
    ],
    "retrieval augmented": [
        "retrieval-augmented generation", "dense retrieval", "vector database",
        "reranking", "hallucination mitigation", "knowledge grounding",
        "open-domain question answering",
    ],
    "multimodal": [
        "vision-language model", "clip", "contrastive language-image pretraining",
        "visual instruction tuning", "image captioning", "text-to-image",
        "flamingo", "llava", "multimodal large language model",
    ],
    "reinforcement learning": [
        "proximal policy optimization", "ppo", "offline reinforcement learning",
        "reward modeling", "rlhf", "decision transformer", "world model",
        "q-learning",
    ],
    "vision transformer": [
        "vit", "self-attention", "swin transformer", "masked image modeling",
        "mae", "contrastive learning", "simclr", "dino",
    ],
    "agent": [
        "tool use", "function calling", "react", "llm agent planning",
        "multi-agent systems", "reflexion", "code generation agent",
    ],
    "model compression": [
        "pruning", "quantization", "knowledge distillation", "low-rank adaptation",
        "awq", "gptq", "qlora", "structured pruning",
    ],
    "interpretability": [
        "mechanistic interpretability", "probing classifiers",
        "sparse autoencoder interpretability", "activation patching",
        "attention analysis", "circuit discovery",
    ],
    "graph neural": [
        "graph neural network", "graph convolutional network", "message passing",
        "graph attention network", "link prediction", "graph transformer",
    ],
    "speech": [
        "whisper", "asr", "automatic speech recognition", "text-to-speech",
        "voice conversion", "audio language model",
    ],
    "video generation": [
        "video diffusion model", "text-to-video", "temporal consistency",
        "video prediction", "video transformer",
    ],
    "alignment": [
        "constitutional ai", "red teaming", "jailbreak defense",
        "direct preference optimization", "dpo", "reward hacking", "safety fine-tuning",
    ],
    "segmentation": [
        "segment anything", "sam", "semantic segmentation",
        "instance segmentation", "medical image segmentation", "promptable segmentation",
    ],
    "federated": [
        "federated learning", "federated averaging", "differential privacy",
        "client heterogeneity", "communication-efficient training",
    ],
    "fine-tuning": [
        "parameter-efficient fine-tuning", "lora", "adapter", "prefix tuning",
        "prompt tuning", "qlora",
    ],
    "time series": [
        "time series forecasting", "temporal fusion transformer",
        "forecasting foundation model", "anomaly detection",
    ],
    "recommender": [
        "collaborative filtering", "recommender system",
        "sequential recommendation", "contrastive recommendation",
    ],
}

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_S2_SEARCH_FIELDS = (
    "paperId,title,authors,abstract,year,venue,publicationVenue,citationCount,"
    "influentialCitationCount,externalIds,openAccessPdf,fieldsOfStudy,"
    "publicationDate,tldr"
)
_S2_ENRICH_FIELDS = (
    "paperId,title,venue,year,citationCount,influentialCitationCount,"
    "externalIds,openAccessPdf,publicationDate"
)

_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1"


class Searcher:
    """Cross-source literature searcher with query expansion and filters.

    Args:
        config: Skill configuration.
        http: Shared HTTP client.
        cache: Optional disk cache.
        llm: Optional LLM backend used for query expansion.
    """

    def __init__(
        self,
        config: MLPaperAnalystConfig,
        http: HTTPClient,
        cache: Optional[DiskCache] = None,
        llm: Optional[LLMBackend] = None,
    ) -> None:
        self._config = config
        self._http = http
        self._cache = cache
        self._llm = llm

    # ------------------------------------------------------------------ #
    # Query expansion                                                    #
    # ------------------------------------------------------------------ #
    async def expand_query(self, intent: str) -> SearchQuery:
        """Expand a natural-language research intent into search terms.

        Uses the LLM when available, otherwise the curated taxonomy.

        Args:
            intent: Raw user intent (e.g. ``diffusion text to 3d``).

        Returns:
            A :class:`SearchQuery` with expanded terms and final query.
        """
        expanded: List[str] = []
        method = "taxonomy"
        if self._llm is not None:
            try:
                raw = await self._llm.complete(
                    system=QUERY_EXPANSION_SYSTEM_PROMPT,
                    user=QUERY_EXPANSION_USER_TEMPLATE.format(intent=intent),
                    temperature=0.3,
                    max_tokens=500,
                )
                data = extract_json_block(raw)
                terms = data.get("expanded_terms", []) if data else []
                expanded = [
                    str(t).strip() for t in terms
                    if isinstance(t, str) and t.strip()
                ][:12]
                method = "llm"
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning("LLM query expansion failed: %s", exc)
        if not expanded:
            expanded = self._taxonomy_expand(intent)
        expanded = self._dedupe_terms(intent, expanded)[:10]
        quoted = [f'"{intent}"'] + [f'"{t}"' for t in expanded]
        return SearchQuery(
            raw_intent=intent,
            expanded_terms=expanded,
            final_query=" OR ".join(quoted),
            expansion_method=method if expanded else "none",
        )

    @staticmethod
    def _dedupe_terms(intent: str, terms: List[str]) -> List[str]:
        """Drop terms duplicating the intent or each other (case-insensitive).

        Args:
            intent: Original intent.
            terms: Candidate expanded terms.

        Returns:
            De-duplicated term list preserving order.
        """
        seen: Set[str] = {intent.lower()}
        out: List[str] = []
        for term in terms:
            key = term.lower()
            if key in seen or key in intent.lower() or intent.lower() in key:
                continue
            seen.add(key)
            out.append(term)
        return out

    @staticmethod
    def _taxonomy_expand(intent: str) -> List[str]:
        """Expand the intent using the curated ML taxonomy.

        Args:
            intent: Raw user intent.

        Returns:
            Up to 10 related keywords from matching sub-field groups.
        """
        lowered = intent.lower()
        matched_groups: List[Tuple[float, List[str]]] = []
        for trigger, terms in QUERY_EXPANSION_TAXONOMY.items():
            trigger_norm = trigger.replace("-", " ")
            if trigger_norm in lowered or any(
                t.lower() in lowered for t in terms[:5]
            ):
                matched_groups.append((float(len(trigger_norm)), terms))
        matched_groups.sort(key=lambda pair: -pair[0])
        expanded: List[str] = []
        seen: Set[str] = set()
        for _, terms in matched_groups:
            for term in terms:
                if term.lower() not in seen and term.lower() not in lowered:
                    seen.add(term.lower())
                    expanded.append(term)
        return expanded[:10]

    # ------------------------------------------------------------------ #
    # Source-specific searches                                           #
    # ------------------------------------------------------------------ #
    def _search_cache_key(self, query: SearchQuery, per_source: int) -> str:
        """Build the disk-cache key for a search."""
        raw = json.dumps(
            {
                "intent": query.raw_intent,
                "expanded": query.expanded_terms,
                "years": query.years,
                "venues": query.venue_filter,
                "min_citations": query.min_citations,
                "sources": query.sources,
                "per_source": per_source,
            },
            sort_keys=True,
        )
        return f"search:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    async def _search_arxiv(
        self, query: SearchQuery, per_source: int
    ) -> List[PaperMetadata]:
        """Search the arXiv Atom API.

        Args:
            query: Expanded query.
            per_source: Maximum entries to request.

        Returns:
            Parsed paper metadata list.

        Raises:
            aiohttp.ClientError: On network failure after retries.
        """
        terms = [query.raw_intent] + query.expanded_terms[:6]
        clauses = [f'all:"{term}"' for term in dict.fromkeys(terms)]
        search = "(" + " OR ".join(clauses) + ")"
        if self._config.arxiv_categories:
            cats = " OR ".join(
                f"cat:{c}" for c in self._config.arxiv_categories
            )
            search = f"({cats}) AND {search}"
        if query.years:
            lo, hi = min(query.years), max(query.years)
            search = f"submittedDate:[{lo}01010000 TO {hi}12312359] AND {search}"
        url = (
            f"{_ARXIV_API}?search_query={quote_plus(search)}"
            f"&sortBy={self._config.arxiv_sort_by}&sortOrder=descending"
            f"&start=0&max_results={min(per_source, 100)}"
        )
        xml_text = await self._http.get_text(url)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"arXiv returned unparseable XML: {exc}") from exc
        papers = []
        for entry in root.findall(f"{_ATOM}entry"):
            try:
                papers.append(self._arxiv_entry_to_metadata(entry))
            except Exception as exc:  # noqa: BLE001 - skip malformed entries
                logger.debug("Skipping malformed arXiv entry: %s", exc)
        return papers

    @staticmethod
    def _arxiv_entry_to_metadata(entry: ET.Element) -> PaperMetadata:
        """Convert one arXiv Atom ``<entry>`` into :class:`PaperMetadata`.

        Args:
            entry: Parsed XML element.

        Returns:
            Paper metadata record.
        """
        id_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        match = re.search(r"arxiv\.org/abs/(.+)$", id_url)
        arxiv_full = match.group(1) if match else ""
        arxiv_id = re.sub(r"v\d+$", "", arxiv_full) or None
        title = _clean_ws(entry.findtext(f"{_ATOM}title") or "")
        abstract = _clean_ws(entry.findtext(f"{_ATOM}summary") or "")
        authors = [
            _clean_ws(a.findtext(f"{_ATOM}name") or "")
            for a in entry.findall(f"{_ATOM}author")
        ]
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf_url: Optional[str] = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                pdf_url = link.get("href")
        doi = (entry.findtext(f"{_ARXIV_NS}doi") or "").strip() or None
        comment = _clean_ws(entry.findtext(f"{_ARXIV_NS}comment") or "") or None
        journal_ref = (
            _clean_ws(entry.findtext(f"{_ARXIV_NS}journal_ref") or "") or None
        )
        categories = [
            c.get("term") for c in entry.findall(f"{_ATOM}category") if c.get("term")
        ]
        venue = journal_ref or _venue_from_comment(comment)
        if arxiv_id is None:
            raise ValueError(f"cannot parse arXiv id from {id_url!r}")
        return PaperMetadata(
            paper_id=arxiv_id,
            source="arxiv",
            title=title or arxiv_id,
            authors=[a for a in authors if a],
            abstract=abstract,
            arxiv_id=arxiv_id,
            doi=doi,
            year=year,
            published_date=published or None,
            venue=venue,
            pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
            abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            html_url=f"https://arxiv.org/html/{arxiv_id}",
            comment=comment,
            keywords=categories,
        )

    @staticmethod
    def _s2_to_metadata(data: Dict[str, Any]) -> PaperMetadata:
        """Convert one Semantic Scholar record into :class:`PaperMetadata`.

        Args:
            data: Raw S2 JSON object.

        Returns:
            Paper metadata record.
        """
        ext = data.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        doi = ext.get("DOI")
        paper_id = str(data.get("paperId") or arxiv_id or doi or "")[:64]
        venue = data.get("venue") or ""
        if not venue:
            pub_venue = data.get("publicationVenue") or {}
            venue = pub_venue.get("name") or ""
        oa_pdf = (data.get("openAccessPdf") or {}).get("url")
        tldr = (data.get("tldr") or {}).get("text")
        published = data.get("publicationDate")
        title = _clean_ws(data.get("title") or "")
        return PaperMetadata(
            paper_id=paper_id or title[:40],
            source="semantic_scholar",
            title=title,
            authors=[
                a.get("name", "")
                for a in (data.get("authors") or [])
                if a.get("name")
            ],
            abstract=_clean_ws(data.get("abstract") or ""),
            arxiv_id=arxiv_id,
            doi=doi,
            year=data.get("year"),
            published_date=published,
            venue=venue or None,
            citation_count=data.get("citationCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            fields_of_study=list(data.get("fieldsOfStudy") or []),
            tldr=tldr,
            pdf_url=(
                f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else oa_pdf
            ),
            abs_url=(
                f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                else (f"https://doi.org/{doi}" if doi else None)
            ),
            html_url=(
                f"https://arxiv.org/html/{arxiv_id}" if arxiv_id else None
            ),
            open_access_pdf_url=oa_pdf,
        )

    async def _search_semantic_scholar(
        self, query: SearchQuery, per_source: int
    ) -> List[PaperMetadata]:
        """Search the Semantic Scholar Graph API with pagination.

        Args:
            query: Expanded query.
            per_source: Maximum records to collect.

        Returns:
            Parsed paper metadata list.

        Raises:
            aiohttp.ClientError: On network failure after retries.
        """
        params: Dict[str, Any] = {
            "query": query.raw_intent,
            "limit": 100,
            "offset": 0,
            "fields": _S2_SEARCH_FIELDS,
        }
        if query.years:
            params["year"] = f"{min(query.years)}-{max(query.years)}"
        collected: List[PaperMetadata] = []
        while len(collected) < per_source:
            data = await self._http.get_json(
                f"{_S2_API}/paper/search", params=params
            )
            batch = data.get("data") or []
            for item in batch:
                try:
                    collected.append(self._s2_to_metadata(item))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Skipping malformed S2 record: %s", exc)
            nxt = data.get("next")
            if not nxt or not batch:
                break
            params["offset"] = nxt
        return collected[:per_source]

    async def _search_huggingface_daily(
        self, query: SearchQuery, per_source: int
    ) -> List[PaperMetadata]:
        """Search Hugging Face Daily Papers (trending/community-curated lens).

        The endpoint returns recent daily papers; results are filtered
        client-side by term overlap with the query.

        Args:
            query: Expanded query.
            per_source: Maximum records to return.

        Returns:
            Papers matching the topic; possibly empty on API mismatch.
        """
        url = f"https://huggingface.co/api/daily_papers?search={quote_plus(query.raw_intent)}"
        data = await self._http.get_json(url)
        if not isinstance(data, list):
            logger.warning("Unexpected HF daily papers response shape; skipped")
            return []
        terms = {query.raw_intent.lower()} | {
            t.lower() for t in query.expanded_terms
        }
        scored: List[Tuple[int, PaperMetadata]] = []
        for item in data:
            paper = item.get("paper", item) if isinstance(item, dict) else None
            if not isinstance(paper, dict):
                continue
            title = _clean_ws(paper.get("title") or "")
            summary = _clean_ws(paper.get("summary") or "")
            haystack = f"{title} {summary}".lower()
            score = sum(1 for t in terms if t and t in haystack)
            if score == 0 or not title:
                continue
            arxiv_id = str(paper.get("id") or paper.get("arxiv_id") or "") or None
            published = item.get("publishedAt") or paper.get("publishedAt")
            year = int(published[:4]) if published and published[:4].isdigit() else None
            authors = [
                a.get("name", "")
                for a in (paper.get("authors") or [])
                if isinstance(a, dict) and a.get("name")
            ]
            scored.append((
                score * 1000 + int(item.get("upvotes", 0) or 0),
                PaperMetadata(
                    paper_id=arxiv_id or title[:40],
                    source="huggingface_daily",
                    title=title,
                    authors=authors,
                    abstract=summary,
                    arxiv_id=arxiv_id,
                    year=year,
                    published_date=published,
                    pdf_url=(
                        f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
                    ),
                    abs_url=(
                        f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
                    ),
                    keywords=["huggingface-daily"],
                ),
            ))
        scored.sort(key=lambda pair: -pair[0])
        return [md for _, md in scored[:per_source]]

    async def _search_paperswithcode(
        self, query: SearchQuery, per_source: int
    ) -> List[PaperMetadata]:
        """Search PapersWithCode.

        Note: PapersWithCode was deprecated/shut down in 2025; this source is
        kept for backward compatibility and fails gracefully.

        Args:
            query: Expanded query.
            per_source: Maximum records to return.

        Returns:
            Papers (usually empty since the service sunset).
        """
        url = (
            "https://paperswithcode.com/api/v1/search/"
            f"?q={quote_plus(query.raw_intent)}&page=1&per_page={min(per_source, 50)}"
        )
        data = await self._http.get_json(url)
        results = data.get("results") or [] if isinstance(data, dict) else []
        papers: List[PaperMetadata] = []
        for item in results:
            paper = item.get("paper", item)
            if not isinstance(paper, dict):
                continue
            arxiv_id = paper.get("arxiv_id") or None
            title = _clean_ws(paper.get("title") or "")
            if not title:
                continue
            published = paper.get("published") or ""
            papers.append(PaperMetadata(
                paper_id=arxiv_id or title[:40],
                source="paperswithcode",
                title=title,
                authors=list(paper.get("authors") or []),
                abstract=_clean_ws(paper.get("abstract") or ""),
                arxiv_id=arxiv_id,
                year=(
                    int(published[:4]) if published[:4].isdigit() else None
                ),
                pdf_url=paper.get("url_pdf"),
                abs_url=paper.get("url_abs"),
            ))
        return papers

    # ------------------------------------------------------------------ #
    # Enrichment, filtering, ranking                                     #
    # ------------------------------------------------------------------ #
    async def _enrich_with_s2(
        self, papers: List[PaperMetadata]
    ) -> List[PaperMetadata]:
        """Fill citation counts / venues using the S2 batch endpoint.

        Args:
            papers: Papers possibly lacking citation/venue data.

        Returns:
            New list with enriched records (originals untouched on failure).
        """
        ids: List[Optional[str]] = []
        for p in papers:
            if p.arxiv_id:
                ids.append(f"ARXIV:{p.arxiv_id}")
            elif p.doi:
                ids.append(f"DOI:{p.doi}")
            elif p.source == "semantic_scholar" and p.paper_id:
                ids.append(p.paper_id)
            else:
                ids.append(None)
        if not any(ids):
            return papers
        chunk: List[str] = [i for i in ids if i]
        try:
            results = await self._http.post_json(
                f"{_S2_API}/paper/batch?fields={_S2_ENRICH_FIELDS}",
                json_body={"ids": chunk[:100]},
            )
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            logger.warning("S2 enrichment failed: %s", exc)
            return papers
        if not isinstance(results, list):
            return papers
        by_s2_id = {
            (r or {}).get("paperId"): r for r in results if isinstance(r, dict)
        }
        enriched: List[PaperMetadata] = []
        cursor = 0
        for paper, identifier in zip(papers, ids):
            if identifier is None:
                enriched.append(paper)
                continue
            record = results[cursor] if cursor < len(results) else None
            cursor += 1
            if not isinstance(record, dict):
                enriched.append(paper)
                continue
            snapshot = by_s2_id.get(record.get("paperId"), record)
            try:
                s2_md = self._s2_to_metadata(snapshot)
                enriched.append(paper.merged_with(s2_md))
            except Exception:  # noqa: BLE001
                enriched.append(paper)
        return enriched

    @staticmethod
    def _dedupe(papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """De-duplicate papers across sources by arXiv id / DOI / title.

        Args:
            papers: Merged multi-source results.

        Returns:
            Deduplicated list preserving first occurrence order; records from
            different sources covering the same paper are merged.
        """
        seen: Dict[str, PaperMetadata] = {}
        for paper in papers:
            key = paper.dedupe_key()
            if key in seen:
                seen[key] = seen[key].merged_with(paper)
            else:
                seen[key] = paper
        return list(seen.values())

    @staticmethod
    def _passes_filters(
        paper: PaperMetadata,
        years: Optional[List[int]],
        venue_filter: Optional[List[str]],
        min_citations: int,
    ) -> bool:
        """Apply year / venue / citation filters to one paper.

        Papers with unknown year are dropped when a year filter is active;
        papers with unknown citation counts are dropped when a citation
        threshold is set.

        Args:
            paper: Candidate paper.
            years: Allowed year range (min/max of the list).
            venue_filter: Allowed venues.
            min_citations: Minimum citation count.

        Returns:
            Whether the paper passes.
        """
        if years:
            if paper.year is None:
                return False
            if not (min(years) <= paper.year <= max(years)):
                return False
        if not venue_matches(paper, venue_filter or []):
            return False
        if min_citations > 0:
            if paper.citation_count is None:
                return False
            if paper.citation_count < min_citations:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Public orchestration                                               #
    # ------------------------------------------------------------------ #
    async def search(
        self,
        intent: str,
        max_results: int = 10,
        years: Optional[List[int]] = None,
        venue_filter: Optional[List[str]] = None,
        min_citations: int = 0,
        sources: Optional[List[str]] = None,
        expand: bool = True,
    ) -> Tuple[List[PaperMetadata], SearchQuery]:
        """Run the full cross-source search pipeline.

        Args:
            intent: Natural-language research intent.
            max_results: Final number of papers to return.
            years: Year filter (list is interpreted as a min-max range).
            venue_filter: Venue names / aliases to restrict to.
            min_citations: Minimum citation threshold (needs S2 enrichment).
            sources: Subset of sources to query (default from config).
            expand: Whether to run query expansion.

        Returns:
            Tuple ``(papers, query)`` where ``papers`` is the filtered and
            ranked result list.
        """
        query = (
            await self.expand_query(intent) if expand
            else SearchQuery(
                raw_intent=intent,
                final_query=f'"{intent}"',
                expansion_method="none",
            )
        )
        query.years = years
        query.venue_filter = venue_filter
        query.min_citations = min_citations
        active = [
            s for s in (sources or self._config.default_sources)
            if s in SUPPORTED_SOURCES
        ]
        query.sources = active

        per_source = min(
            self._config.per_source_limit_cap,
            max(max_results * self._config.fetch_extra_factor, max_results),
        )

        cache_key = self._search_cache_key(query, per_source)
        cached = self._cache.get(cache_key) if self._cache else None
        if cached:
            try:
                return (
                    [PaperMetadata.model_validate(item) for item in cached],
                    query,
                )
            except ValidationError as exc:
                logger.debug("Ignoring corrupt cached search: %s", exc)

        dispatch: Dict[str, Callable[[SearchQuery, int], Awaitable[List[PaperMetadata]]]] = {
            "arxiv": self._search_arxiv,
            "semantic_scholar": self._search_semantic_scholar,
            "huggingface_daily": self._search_huggingface_daily,
            "paperswithcode": self._search_paperswithcode,
        }
        outcomes = await asyncio.gather(
            *(dispatch[name](query, per_source) for name in active),
            return_exceptions=True,
        )
        merged: List[PaperMetadata] = []
        for name, outcome in zip(active, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning("Source '%s' failed: %s", name, outcome)
                continue
            merged.extend(outcome)

        papers = self._dedupe(merged)
        if (venue_filter or min_citations > 0) and papers:
            papers = await self._enrich_with_s2(papers)
        papers = [
            p for p in papers
            if self._passes_filters(p, years, venue_filter, min_citations)
        ]
        papers.sort(key=lambda p: (p.citation_count or 0, p.year or 0),
                    reverse=True)
        papers = papers[:max_results]

        if self._cache and papers:
            self._cache.set(cache_key, [p.model_dump(mode="json") for p in papers])
        return papers, query


# --------------------------------------------------------------------------- #
# Section canonicalisation (shared by HTML and PDF parsing)                   #
# --------------------------------------------------------------------------- #

_SECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^abstract\b"), "abstract"),
    (re.compile(r"^(introduction|overview|motivation)\b"), "introduction"),
    (re.compile(
        r"^(related work|prior work|literature review|background and related)"
    ), "related_work"),
    (re.compile(r"^(background|preliminar(y|ies)|notations?)\b"), "background"),
    (re.compile(
        r"^((our|proposed|the)\s+)?"
        r"(method(s|ology)?|approach|model|framework|architecture|system|"
        r"pipeline|formulation|system (overview|design)|"
        r"problem (formulation|setting|definition))\b"
    ), "method"),
    (re.compile(
        r"^(implementation( details)?|training( details| procedure)?|"
        r"experimental (setup|settings?|protocol)|setup|dataset s?(and setup)?|"
        r"datasets?( and (setup|implementation|preprocessing))?|"
        r"experiments?( setup)?|evaluation( (setup|protocol))?)\b"
    ), "experiments"),
    (re.compile(
        r"^(results?( and (discussion|analysis))?|ablations?( stud(y|ies))?|"
        r"experimental results?|main results|evaluation results|"
        r"quantitative (results|evaluation|analysis)|qualitative (results|"
        r"evaluation|analysis)|comparisons?( with (existing|related|prior|"
        r"state-?of-?-the-?-art) (methods?|works?))?|analysis)\b"
    ), "results"),
    (re.compile(r"^(discussion|analysis and discussion|remarks)\b"), "discussion"),
    (re.compile(r"^(limitations?|threats to validity|weaknesses)\b"), "limitations"),
    (re.compile(
        r"^(conclusions?( and (future work|outlook))?|summary and (conclusions?|"
        r"outlook)|concluding remarks|takeaways?|final remarks)\b"
    ), "conclusion"),
    (re.compile(
        r"^(acknowledge?ments?|funding|ethics( (statement|considerations))?|"
        r"broader impact|reproducibility( statement)?)\b"
    ), "acknowledgments"),
    (re.compile(r"^(references|bibliography|works cited)\b"), "references"),
]

_NUMBERING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]{1,5})\.?\s+")


def strip_section_numbering(title: str) -> str:
    """Remove leading numbering from a section title.

    Handles ``"1 Introduction"``, ``"3.2. Method"`` and ``"IV. RESULTS"``.

    Args:
        title: Raw section heading.

    Returns:
        Title without the numeric/roman prefix.
    """
    return _NUMBERING_RE.sub("", (title or "").strip()).strip()


def canonicalize_section(title: str) -> Optional[str]:
    """Map a section title to its canonical pipeline name.

    Args:
        title: Raw section heading (numbering allowed).

    Returns:
        One of ``abstract/introduction/related_work/background/method/
        experiments/results/discussion/limitations/conclusion/
        acknowledgments/references`` or ``None`` when unmapped.
    """
    cleaned = strip_section_numbering(title).lower().rstrip(":.")
    if not cleaned:
        return None
    for pattern, canonical in _SECTION_PATTERNS:
        if pattern.match(cleaned):
            return canonical
    return None


# --------------------------------------------------------------------------- #
# Module 2 — Extractor                                                        #
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - optional dependency guard
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:  # pragma: no cover - prefer fast parser when installed
    import lxml  # noqa: F401
    _BS_PARSER = "lxml"
except ImportError:  # pragma: no cover
    _BS_PARSER = "html.parser"


class Extractor:
    """Full-text acquisition and structured section extraction.

    Strategy (in order): ar5iv HTML → arXiv native HTML → official PDF via
    PyMuPDF → metadata-only fallback. All results are disk-cached.

    Args:
        config: Skill configuration.
        http: Shared HTTP client.
        cache: Optional disk cache.
    """

    def __init__(
        self,
        config: MLPaperAnalystConfig,
        http: HTTPClient,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self._config = config
        self._http = http
        self._cache = cache

    # ------------------------------------------------------------------ #
    # Identifier resolution                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_arxiv_id(value: str) -> Optional[str]:
        """Extract a bare arXiv id from an id or URL string.

        Args:
            value: User-provided identifier or URL.

        Returns:
            Versionless arXiv id (e.g. ``2312.06621``) or ``None``.
        """
        value = (value or "").strip()
        match = re.search(
            r"arxiv\.org/(?:abs|pdf|html)/(?:v\d+/)?(\d{4}\.\d{4,5})(?:v\d+)?",
            value,
        ) or re.search(
            r"ar5iv\.labs\.arxiv\.org/html/(\d{4}\.\d{4,5})(?:v\d+)?", value
        )
        if match:
            return match.group(1)
        if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", value):
            return re.sub(r"v\d+$", "", value)
        return None

    async def _metadata_from_s2(
        self, s2_id: Optional[str] = None, arxiv_id: Optional[str] = None,
        doi: Optional[str] = None,
    ) -> PaperMetadata:
        """Fetch metadata for one paper from the S2 single-paper endpoint.

        Args:
            s2_id: Semantic Scholar paper id.
            arxiv_id: arXiv id.
            doi: DOI.

        Returns:
            Paper metadata.

        Raises:
            ValueError: When no identifier is supplied.
            aiohttp.ClientError: On network failure.
        """
        if arxiv_id:
            path = f"arXiv:{arxiv_id}"
        elif doi:
            path = f"DOI:{doi}"
        elif s2_id:
            path = s2_id
        else:
            raise ValueError("no identifier supplied")
        data = await self._http.get_json(
            f"{_S2_API}/paper/{path}", params={"fields": _S2_SEARCH_FIELDS}
        )
        return Searcher._s2_to_metadata(data)

    async def _metadata_from_arxiv_id(self, arxiv_id: str) -> PaperMetadata:
        """Fetch metadata via the arXiv ``id_list`` API endpoint.

        Args:
            arxiv_id: Versionless arXiv id.

        Returns:
            Paper metadata.

        Raises:
            ValueError: When the id is unknown to arXiv.
        """
        url = f"{_ARXIV_API}?id_list={quote_plus(arxiv_id)}&max_results=1"
        xml_text = await self._http.get_text(url)
        root = ET.fromstring(xml_text)
        entry = root.find(f"{_ATOM}entry")
        if entry is None:
            raise ValueError(f"arXiv id {arxiv_id} not found")
        return Searcher._arxiv_entry_to_metadata(entry)

    async def _resolve(self, target: Union[str, PaperMetadata]) -> PaperMetadata:
        """Resolve any supported identifier/URL into paper metadata.

        Supported inputs: arXiv id (``2312.06621``, with version), arXiv
        abs/pdf/html/ar5iv URLs, bare DOI, S2 paper id (40-hex), or an
        existing :class:`PaperMetadata`.

        Args:
            target: Identifier, URL or metadata object.

        Returns:
            Resolved paper metadata.

        Raises:
            ValueError: When the input matches no known format.
        """
        if isinstance(target, PaperMetadata):
            return target
        raw = (target or "").strip()
        if not raw:
            raise ValueError("empty paper identifier")

        arxiv_id = self.parse_arxiv_id(raw)
        if arxiv_id:
            try:
                return await self._metadata_from_s2(arxiv_id=arxiv_id)
            except Exception as exc:  # noqa: BLE001 - fall back to arXiv API
                logger.debug("S2 lookup failed for %s (%s); using arXiv API",
                             arxiv_id, exc)
                return await self._metadata_from_arxiv_id(arxiv_id)

        doi_match = re.search(r"(10\.\d{4,9}/\S+)", raw)
        if doi_match:
            return await self._metadata_from_s2(doi=doi_match.group(1))

        if re.fullmatch(r"[0-9a-f]{40}", raw):
            return await self._metadata_from_s2(s2_id=raw)

        raise ValueError(
            "Unrecognised paper identifier. Use an arXiv id (2312.06621), an "
            "arXiv/DOI URL, a bare DOI, or a 40-char Semantic Scholar id."
        )

    # ------------------------------------------------------------------ #
    # HTML extraction (ar5iv / arXiv HTML, both LaTeXML-generated)       #
    # ------------------------------------------------------------------ #
    def _parse_latexml(
        self, html: str, metadata: PaperMetadata, source: str
    ) -> PaperContent:
        """Parse LaTeXML-generated paper HTML into structured content.

        Args:
            html: Raw HTML document.
            metadata: Paper metadata for the document.
            source: ``ar5iv`` or ``arxiv_html``.

        Returns:
            Structured paper content (may be empty on parse failure).
        """
        soup = BeautifulSoup(html, _BS_PARSER)
        root = soup.find("article") or soup.body or soup
        cfg = self._config
        warnings: List[str] = []
        sections: Dict[str, str] = {}
        extra: Dict[str, str] = {}

        affiliations: List[str] = []
        for el in soup.select(".ltx_role_affiliation"):
            text = _clean_ws(el.get_text(" ", strip=True))
            if text and text not in affiliations:
                affiliations.append(text)
            if len(affiliations) >= 8:
                break

        abstract_el = soup.select_one("div.ltx_abstract") or soup.select_one(
            "section.ltx_abstract"
        )
        abstract = metadata.abstract or ""
        if abstract_el is not None:
            abstract = re.sub(
                r"^abstract\s*", "",
                _clean_ws(abstract_el.get_text(" ", strip=True)),
                flags=re.IGNORECASE,
            )
        if abstract:
            sections["abstract"] = abstract[: cfg.max_section_chars]

        for sec in root.find_all("section"):
            if sec.find_parent("section") is not None:
                continue  # nested subsection — text is included by the parent
            classes = set(sec.get("class") or [])
            body = _clean_ws(sec.get_text(" ", strip=True))
            if "ltx_bibliography" in classes:
                sections["references"] = body[:1500]
                continue
            if "ltx_abstract" in classes:
                continue
            if "ltx_appendix" in classes:
                extra["appendix"] = body[:4000]
                continue
            heading = sec.find(["h2", "h3", "h4", "h5", "h6"])
            title = _clean_ws(heading.get_text(" ", strip=True)) if heading else ""
            text = body
            if title and text.startswith(title):
                text = text[len(title):].strip()
            if len(text) < 40:
                continue
            canonical = canonicalize_section(title or "")
            if canonical:
                existing = sections.get(canonical, "")
                merged = f"{existing} {text}".strip()
                sections[canonical] = merged[: cfg.max_section_chars]
            elif title:
                extra[_slugify(title)] = text[: cfg.max_section_chars]

        figures: List[FigureInfo] = []
        seen_caps: Set[str] = set()
        for kind, selector in (("figure", "figure.ltx_figure"),
                               ("table", "figure.ltx_table")):
            for fig in soup.select(selector):
                caption_el = fig.select_one("figcaption")
                if caption_el is None:
                    continue
                caption = _clean_ws(caption_el.get_text(" ", strip=True))
                if not caption or caption in seen_caps:
                    continue
                seen_caps.add(caption)
                id_match = re.match(r"(Figure|Fig\.?|Table)\s*([\d\.]+)", caption)
                figure_id = (
                    f"{kind.title()} {id_match.group(2)}"
                    if id_match else f"{kind.title()} {len(figures) + 1}"
                )
                figures.append(FigureInfo(
                    figure_id=figure_id, caption=caption[:600], kind=kind
                ))
                if len(figures) >= cfg.max_figures:
                    break
            if len(figures) >= cfg.max_figures:
                break

        algorithms: List[AlgorithmInfo] = []
        for alg in soup.select("figure.ltx_algorithm, div.ltx_algorithm"):
            caption_el = alg.select_one("figcaption")
            caption = (
                _clean_ws(caption_el.get_text(" ", strip=True))
                if caption_el else ""
            )
            body = _clean_ws(alg.get_text(" ", strip=True))
            if caption and body.startswith(caption):
                body = body[len(caption):].strip()
            id_match = re.match(r"Algorithm\s*([\d\.]+)", caption)
            algorithms.append(AlgorithmInfo(
                algorithm_id=f"Algorithm {id_match.group(1) if id_match else len(algorithms) + 1}",
                caption=caption[:300],
                pseudocode=body[:1500],
            ))
            if len(algorithms) >= cfg.max_algorithms:
                break

        formulas: List[FormulaInfo] = []
        for math_el in soup.find_all("math", attrs={"alttext": True}):
            display = math_el.get("display") == "block"
            parent_classes = " ".join(
                math_el.parent.get("class") or []) if math_el.parent else ""
            if not display and "ltx_equation" not in parent_classes:
                continue
            latex = _clean_ws(math_el.get("alttext", ""))
            if not latex or len(latex) < 3:
                continue
            context_el = math_el.find_parent("p") or math_el.parent
            context = _clean_ws(
                context_el.get_text(" ", strip=True) if context_el else ""
            )[:180]
            formulas.append(FormulaInfo(latex=latex[:400], context=context))
            if len(formulas) >= cfg.max_formulas:
                break

        if not sections and not extra:
            warnings.append("HTML parse produced no sections")
        return PaperContent(
            metadata=metadata,
            sections=sections,
            extra_sections=extra,
            affiliations=affiliations,
            figures=figures,
            algorithms=algorithms,
            formulas=formulas,
            extraction_source=source,  # type: ignore[arg-type]
            parse_warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # PDF fallback                                                       #
    # ------------------------------------------------------------------ #
    def _parse_pdf(self, data: bytes, metadata: PaperMetadata) -> PaperContent:
        """Parse a paper PDF with PyMuPDF into structured content.

        Args:
            data: Raw PDF bytes.
            metadata: Paper metadata.

        Returns:
            Structured content extracted via heading heuristics.

        Raises:
            RuntimeError: When PyMuPDF is not installed or the PDF is broken.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed; cannot parse PDF")
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            text = "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()
        text = text[: self._config.max_pdf_chars]

        sections: Dict[str, List[str]] = {"other": []}
        extra: Dict[str, List[str]] = {}
        current = "other"
        current_extra: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                sections[current].append("")
                if current_extra:
                    extra[current_extra].append("")
                continue
            heading_target = self._pdf_heading_target(stripped)
            if heading_target is not None:
                current = heading_target
                sections.setdefault(current, [])
                current_extra = None
                continue
            extra_key = self._pdf_extra_heading(stripped)
            if extra_key is not None:
                current = "other"
                sections[current].append("")
                current_extra = extra_key
                extra.setdefault(current_extra, [])
                continue
            sections[current].append(stripped)
            if current_extra:
                extra[current_extra].append(stripped)

        merged: Dict[str, str] = {}
        for name, lines in sections.items():
            body = _clean_ws(" ".join(lines))
            if len(body) > 40:
                merged[name] = body[: self._config.max_section_chars]
        if not merged.get("abstract") and metadata.abstract:
            merged["abstract"] = metadata.abstract[: self._config.max_section_chars]

        figures: List[FigureInfo] = []
        algorithms: List[AlgorithmInfo] = []
        seen: Set[str] = set()
        for m in re.finditer(
            r"(Figure|Fig\.|Table|Algorithm)\s+(\d+)[:.\-–]?\s*([^\n]{0,400})",
            text,
        ):
            kind_raw, num, caption = m.group(1), m.group(2), _clean_ws(m.group(3))
            ident = f"{kind_raw} {num}"
            if ident in seen or not caption:
                continue
            seen.add(ident)
            if kind_raw.startswith("Alg"):
                algorithms.append(AlgorithmInfo(
                    algorithm_id=ident, caption=caption[:300], pseudocode=""
                ))
            else:
                figures.append(FigureInfo(
                    figure_id=ident,
                    caption=caption[:600],
                    kind="table" if kind_raw == "Table" else "figure",
                ))
            if len(figures) >= self._config.max_figures and len(algorithms) >= 1:
                break

        return PaperContent(
            metadata=metadata,
            sections=merged,
            extra_sections={
                k: _clean_ws(" ".join(v))[: self._config.max_section_chars]
                for k, v in extra.items()
            },
            figures=figures[: self._config.max_figures],
            algorithms=algorithms[: self._config.max_algorithms],
            formulas=[],
            full_text=text,
            extraction_source="pdf",
            parse_warnings=["PDF fallback: formulas and affiliations unavailable"],
        )

    @staticmethod
    def _pdf_heading_target(line: str) -> Optional[str]:
        """Return a canonical section name when ``line`` is a numbered heading.

        Args:
            line: A stripped PDF text line.

        Returns:
            Canonical name or ``None``.
        """
        if len(line) > 90:
            return None
        match = re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]{1,5})\.?\s+(.+)$", line)
        if match and canonicalize_section(match.group(1)):
            return canonicalize_section(match.group(1))
        if line.isupper() and len(line.split()) <= 8:
            return canonicalize_section(line)
        if re.match(r"^abstract\b", line, re.IGNORECASE):
            return "abstract"
        return None

    @staticmethod
    def _pdf_extra_heading(line: str) -> Optional[str]:
        """Return a slug for a numbered-but-unmapped level-1 PDF heading.

        Args:
            line: A stripped PDF text line.

        Returns:
            Slug or ``None``.
        """
        match = re.match(r"^(\d+)\.?\s+([A-Z][^.]{2,60})$", line)
        if match and len(line) < 80:
            return _slugify(match.group(2))
        return None

    # ------------------------------------------------------------------ #
    # Orchestration                                                      #
    # ------------------------------------------------------------------ #
    async def fetch_and_parse(
        self, target: Union[str, PaperMetadata]
    ) -> PaperContent:
        """Fetch and parse one paper using the full fallback chain.

        Args:
            target: Identifier / URL / existing metadata.

        Returns:
            Structured content; never raises for recoverable failures — a
            metadata-only content with warnings is returned instead.

        Raises:
            ValueError: When the identifier cannot be resolved at all.
        """
        metadata = await self._resolve(target)
        warnings: List[str] = []

        cache_key = (
            f"content:{metadata.arxiv_id or metadata.doi or metadata.paper_id}"
        )
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None:
            try:
                return PaperContent.model_validate(cached)
            except ValidationError as exc:
                logger.debug("Ignoring corrupt cached content: %s", exc)

        arxiv_id = metadata.arxiv_id
        if arxiv_id:
            routes: List[Tuple[str, str]] = []
            if "ar5iv" in self._config.html_sources:
                routes.append(
                    ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}")
                )
            if "arxiv_html" in self._config.html_sources:
                routes.append(("arxiv_html", f"https://arxiv.org/html/{arxiv_id}"))
            for source, url in routes:
                try:
                    html = await self._http.get_text(url)
                    content = self._parse_latexml(html, metadata, source)
                    if content.has_fulltext or content.sections.get("abstract"):
                        if self._cache:
                            self._cache.set(
                                cache_key,
                                content.model_dump(mode="json"),
                                ttl=self._config.content_cache_ttl,
                            )
                        return content
                    logger.debug("%s parse empty for %s", source, arxiv_id)
                except Exception as exc:  # noqa: BLE001 - try next route
                    logger.debug("HTML route %s failed for %s: %s",
                                 source, arxiv_id, exc)
                    warnings.append(f"{source} unavailable: {type(exc).__name__}")

        if self._config.pdf_fallback_enabled:
            pdf_url = (
                f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id
                else (metadata.pdf_url or metadata.open_access_pdf_url)
            )
            if pdf_url:
                try:
                    pdf_bytes = await self._http.get_bytes(pdf_url)
                    content = self._parse_pdf(pdf_bytes, metadata)
                    content.parse_warnings.extend(warnings)
                    if self._cache:
                        self._cache.set(
                            cache_key,
                            content.model_dump(mode="json"),
                            ttl=self._config.content_cache_ttl,
                        )
                    return content
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF fallback failed for %s: %s",
                                   metadata.paper_id, exc)
                    warnings.append(f"pdf fallback failed: {exc}")

        warnings.append("no full text available; using metadata only")
        content = PaperContent(
            metadata=metadata,
            sections={"abstract": metadata.abstract} if metadata.abstract else {},
            extraction_source="metadata_only",
            parse_warnings=warnings,
        )
        return content


# --------------------------------------------------------------------------- #
# Module 3 — Analyzer                                                         #
# --------------------------------------------------------------------------- #

_KNOWN_DATASETS: List[str] = [
    "ImageNet-1k", "ImageNet-21k", "ImageNet", "ImageNet-A", "ImageNet-R",
    "CIFAR-10", "CIFAR-100", "CIFAR", "MNIST", "Fashion-MNIST", "STL-10",
    "SVHN", "COCO", "ADE20K", "Cityscapes", "PASCAL VOC", "PASCAL-Context",
    "GLUE", "SuperGLUE", "SQuAD 2.0", "SQuAD", "Natural Questions",
    "TriviaQA", "HotpotQA", "DROP", "BoolQ", "MMLU", "GSM8K", "MATH",
    "HumanEval", "MBPP", "BIG-Bench", "HellaSwag", "WinoGrande", "TruthfulQA",
    "PIQA", "ARC-Challenge", "ARC-Easy", "WMT14", "WMT16", "WikiText-103",
    "WikiText-2", "LAMBADA", "C4", "The Pile", "LAION-5B", "LAION-400M",
    "FFHQ", "AFHQ", "CelebA-HQ", "CelebA", "LSUN", "ShapeNet", "DTU",
    "Tanks and Temples", "Mip-NeRF 360", "ScanNet++", "ScanNet", "KITTI",
    "nuScenes", "Waymo Open Dataset", "Argoverse 2", "LibriSpeech",
    "LibriLight", "AudioSet", "ESC-50", "VoxCeleb", "PubMedQA", "MedQA",
    "MIMIC-CXR", "CheXpert", "ISIC 2018", "OGB", "Cora", "Amazon Reviews",
    "Yelp", "MovieLens", "Criteo", "AV-Deepfake", "LONGER",
]

_METRIC_PATTERN = re.compile(
    r"\b(top-?1 accuracy|top-?5 accuracy|accuracy|F1 score|F1|BLEU-\d|BLEU|"
    r"ROUGE-L|ROUGE-?1|ROUGE-?2|ROUGE|METEOR|BERTScore|mIoU|IoU|mAP|FID|"
    r"Inception Score|IS score|CLIP-?\s?[Ss]core|PSNR|SSIM|LPIPS|"
    r"Chamfer [Dd]istance|AUC|AUROC|perplexity|PPL|exact match|EM score|"
    r"pass@\d+|win rate|Elo|WER|CER|EER|BD-rate)\b"
)

_KNOWN_BASELINES: List[str] = [
    "BERT", "RoBERTa", "GPT-2", "GPT-3", "GPT-4", "LLaMA", "Llama 2",
    "Llama 3", "T5", "ResNet-18", "ResNet-50", "ResNet", "ViT", "Swin",
    "DETR", "YOLOv8", "YOLOv5", "YOLO", "Faster R-CNN", "Mask R-CNN",
    "Mask2Former", "U-Net", "CLIP", "Stable Diffusion", "Latent Diffusion",
    "DreamFusion", "Magic3D", "Point-E", "NeRF", "Instant-NGP",
    "3D Gaussian Splatting", "Mip-NeRF 360", "DDPM", "DINO", "DINOv2",
    "MAE", "SimCLR", "BYOL", "SAM", "BLIP-2", "BLIP", "LLaVA", "Flamingo",
    "Whisper", "DPO", "PPO", "SAC", "TD3", "DreamerV3", "ChatGPT",
    "BLOOM", "Falcon", "Mistral", "Mixtral", "Qwen", "PaLM", "Gemini",
]

_CATEGORY_RULES: List[Tuple[str, str]] = [
    (r"diffusion|flow matching|score-based", "Diffusion Generative"),
    (r"reinforcement learning|reward model|policy gradient|\bppo\b|\brlhf\b",
     "Reinforcement Learning"),
    (r"large language model|\bllm\b|autoregressive|next-token|instruction",
     "Autoregressive / LLM"),
    (r"vision-language|multimodal|text-to-image|image-text",
     "Multimodal"),
    (r"graph neural|message passing|graph convolution",
     "Graph Neural Network"),
    (r"contrastive|self-supervised|masked image|mask(ed)? modeling",
     "Self-Supervised"),
    (r"quantiz|pruning|distill|inference accelerat|efficien|lora",
     "Systems & Efficiency"),
    (r"adversarial|\bgan\b", "Adversarial Generative"),
    (r"transformer|attention", "Transformer-based"),
    (r"convolutional|\bcnn\b", "CNN-based"),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, count: int = 1) -> str:
    """Return the first ``count`` sentences of ``text``.

    Args:
        text: Input text.
        count: Number of sentences.

    Returns:
        Joined sentences (may be empty).
    """
    sentences = _SENTENCE_SPLIT.split(_clean_ws(text))
    return " ".join(s for s in sentences[:count] if s)


class _SingleAnalysis(BaseModel):
    """Internal schema validating the single-pass LLM JSON response."""

    core_contribution: CoreContribution = CoreContribution()
    methodology: MethodologyBreakdown = MethodologyBreakdown()
    experiments: ExperimentalValidation = ExperimentalValidation()
    critique: Critique = Critique()
    overall_assessment: str = ""
    confidence: str = "medium"


class Analyzer:
    """Four-aspect deep analysis engine with LLM + heuristic fallback.

    Args:
        config: Skill configuration.
        llm: LLM backend (``None`` forces heuristic mode).
        analysis_mode: Optional override of ``config.analysis_mode``.
    """

    def __init__(
        self,
        config: MLPaperAnalystConfig,
        llm: Optional[LLMBackend] = None,
        analysis_mode: Optional[str] = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._mode = analysis_mode or config.analysis_mode

    # ------------------------------------------------------------------ #
    # LLM plumbing                                                       #
    # ------------------------------------------------------------------ #
    async def _call_json(
        self, user_prompt: str, model_cls: Type[BaseModel]
    ) -> BaseModel:
        """Run one LLM call and validate its JSON against ``model_cls``.

        One repair round-trip is attempted when the output is not valid
        JSON or fails schema validation.

        Args:
            user_prompt: Fully-rendered user prompt.
            model_cls: Pydantic model to validate against.

        Returns:
            Validated model instance.

        Raises:
            RuntimeError: When both attempts fail.
        """
        assert self._llm is not None
        prompt = user_prompt
        last_error: str = ""
        for attempt in range(2):
            raw = await self._llm.complete(
                system=ACADEMIC_ANALYST_SYSTEM_PROMPT,
                user=prompt,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
            data = extract_json_block(raw)
            if data is None:
                last_error = "response contained no JSON object"
            else:
                try:
                    return model_cls.model_validate(data)
                except ValidationError as exc:
                    last_error = str(exc)[:500]
            prompt = (
                f"{user_prompt}\n\nIMPORTANT: your previous response was "
                f"rejected ({last_error}). Output ONLY a single valid JSON "
                f"object matching the schema exactly — no fences, no extra "
                f"keys, no commentary."
            )
        raise RuntimeError(f"LLM JSON parsing failed: {last_error}")

    # ------------------------------------------------------------------ #
    # Heuristic fallback components                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scan_known(text: str, known: List[str]) -> List[str]:
        """Case-sensitive word-boundary scan for known entity names.

        Args:
            text: Haystack text.
            known: Candidate entity names.

        Returns:
            Matched names in haystack order of first appearance.
        """
        found: List[str] = []
        for name in known:
            if re.search(rf"\b{re.escape(name)}\b", text):
                found.append(name)
        return found

    def _heuristic_report(self, content: PaperContent) -> AnalysisReport:
        """Build an extractive analysis without any LLM.

        Args:
            content: Parsed paper content.

        Returns:
            A low-confidence report with explicitly flagged gaps.
        """
        md = content.metadata
        abstract = content.section_text("abstract", default=md.abstract)
        intro = content.section_text("introduction")
        method = content.section_text("method")
        experiments_text = content.section_text(
            "experiments", "results", default=""
        )
        limitations_text = content.section_text("limitations")
        conclusion = content.section_text("conclusion")

        searchable = " ".join(
            [md.title, abstract, intro, method, experiments_text, conclusion]
        )[:40000]
        sentences = [
            s for s in _SENTENCE_SPLIT.split(_clean_ws(searchable)) if s
        ]
        motivation = next(
            (s for s in sentences
             if re.search(r"\bhowever\b|\bbut\b|remains? (?:a )?(?:challenge|problem)|"
                          r"limitation|difficult|fail", s, re.I)),
            _first_sentences(abstract, 2),
        )
        proposal = next(
            (s for s in sentences
             if re.search(r"we (?:propose|present|introduce|develop)", s, re.I)),
            _first_sentences(abstract, 1),
        )
        result_sentence = next(
            (s for s in sentences
             if re.search(r"outperform|improv|surpass|state-of-the-art|SOTA",
                          s, re.I)),
            "",
        )
        category = "Uncategorized"
        for pattern, label in _CATEGORY_RULES:
            if re.search(pattern, searchable, re.I):
                category = label
                break

        limitations = [
            _first_sentences(s, 1)
            for s in _SENTENCE_SPLIT.split(_clean_ws(limitations_text))[:3]
        ] if limitations_text else ["Not analyzed (heuristic mode)"]

        futures = [
            _first_sentences(s, 1) for s in
            _SENTENCE_SPLIT.split(_clean_ws(conclusion))[:6]
            if re.search(r"future|future work|next step|promising", s, re.I)
        ]

        report = AnalysisReport(
            paper_id=md.paper_id,
            arxiv_id=md.arxiv_id,
            doi=md.doi,
            title=md.title,
            authors=list(md.authors),
            year=md.year,
            venue=md.venue,
            citation_count=md.citation_count,
            core_contribution=CoreContribution(
                one_line_summary=_first_sentences(abstract, 1)
                or md.title,
                motivation=motivation or "Not analyzed (heuristic mode)",
                novelty=proposal or "Not analyzed (heuristic mode)",
                pain_points_addressed=[],
            ),
            methodology=MethodologyBreakdown(
                method_category=category,
                architecture=_first_sentences(method, 3)
                or "Not analyzed (heuristic mode)",
                loss_functions="Not analyzed (heuristic mode)",
                training_techniques="Not analyzed (heuristic mode)",
                key_assumptions=[],
            ),
            experiments=ExperimentalValidation(
                datasets=self._scan_known(
                    f"{abstract} {experiments_text}", _KNOWN_DATASETS
                )[:10],
                metrics=sorted(
                    set(m.group(0) for m in
                        _METRIC_PATTERN.finditer(
                            f"{abstract} {experiments_text}")
                        )
                )[:12],
                baselines=self._scan_known(searchable, _KNOWN_BASELINES)[:10],
                sota_improvement=result_sentence
                or "Not analyzed (heuristic mode)",
                main_results_summary=result_sentence
                or "Not analyzed (heuristic mode)",
            ),
            critique=Critique(
                strengths=[
                    s for s in sentences
                    if re.search(r"outperform|advantage|efficient|scalab",
                                 s, re.I)
                ][:2],
                limitations=limitations,
                future_directions=futures
                or ["Not analyzed (heuristic mode)"],
            ),
            overall_assessment=(
                "Heuristic extractive analysis (no LLM configured). "
                f"Core claim: {_first_sentences(abstract, 1) or md.title}"
            ),
            confidence="low",
            generated_by="heuristic-fallback",
        )
        return report

    # ------------------------------------------------------------------ #
    # Main entry                                                         #
    # ------------------------------------------------------------------ #
    async def analyze(self, content: PaperContent) -> AnalysisReport:
        """Produce a four-aspect :class:`AnalysisReport` for one paper.

        With an LLM configured, runs either a single JSON call or a
        four-step prompt chain; any LLM failure falls back to the
        extractive heuristic (unless disabled in config).

        Args:
            content: Parsed paper content.

        Returns:
            The completed analysis report.
        """
        md = content.metadata
        if self._llm is None:
            if not self._config.heuristic_fallback:
                raise RuntimeError(
                    "no LLM backend configured and heuristic_fallback=False"
                )
            return self._heuristic_report(content)

        context = render_analysis_context(
            content, self._config.analyzer_context_char_budget
        )
        model_tag = getattr(self._llm, "model", "custom")
        try:
            if self._mode == "chain":
                return await self._analyze_chain(content, context, model_tag)
            return await self._analyze_single(content, context, model_tag)
        except Exception as exc:  # noqa: BLE001 - resilient pipeline
            logger.warning(
                "LLM analysis failed for %s (%s); heuristic fallback used",
                md.paper_id, exc,
            )
            if not self._config.heuristic_fallback:
                raise
            return self._heuristic_report(content)

    async def _analyze_single(
        self, content: PaperContent, context: str, model_tag: str
    ) -> AnalysisReport:
        """Run the single-pass full analysis prompt.

        Args:
            content: Parsed paper content.
            context: Rendered analysis context.
            model_tag: LLM model identifier for provenance.

        Returns:
            Completed analysis report.

        Raises:
            RuntimeError: On irreparable JSON/validation failure.
        """
        prompt = SINGLE_PASS_USER_TEMPLATE.format(context=context)
        result = await self._call_json(prompt, _SingleAnalysis)
        assert isinstance(result, _SingleAnalysis)
        md = content.metadata
        confidence = result.confidence if result.confidence in {
            "high", "medium", "low"
        } else "medium"
        if not content.has_fulltext and confidence == "high":
            confidence = "medium"
        return AnalysisReport(
            paper_id=md.paper_id,
            arxiv_id=md.arxiv_id,
            doi=md.doi,
            title=md.title,
            authors=list(md.authors),
            year=md.year,
            venue=md.venue,
            citation_count=md.citation_count,
            core_contribution=result.core_contribution,
            methodology=result.methodology,
            experiments=result.experiments,
            critique=result.critique,
            overall_assessment=result.overall_assessment,
            confidence=confidence,  # type: ignore[arg-type]
            generated_by=f"llm:{model_tag}",
        )

    async def _analyze_chain(
        self, content: PaperContent, context: str, model_tag: str
    ) -> AnalysisReport:
        """Run the four-step prompt chain (one call per aspect).

        Args:
            content: Parsed paper content.
            context: Rendered analysis context.
            model_tag: LLM model identifier for provenance.

        Returns:
            Completed analysis report.
        """
        heuristic = self._heuristic_report(content)
        templates = {
            "contribution": CONTRIBUTION_USER_TEMPLATE,
            "methodology": METHODOLOGY_USER_TEMPLATE,
            "experiments": EXPERIMENTS_USER_TEMPLATE,
            "critique": CRITIQUE_USER_TEMPLATE,
        }
        model_classes = {
            "contribution": CoreContribution,
            "methodology": MethodologyBreakdown,
            "experiments": ExperimentalValidation,
            "critique": Critique,
        }
        aspects: Dict[str, BaseModel] = {}
        for aspect, template in templates.items():
            prompt = template.format(context=context)
            try:
                parsed = await self._call_json(prompt, model_classes[aspect])
                aspects[aspect] = parsed
            except Exception as exc:  # noqa: BLE001 - per-aspect fallback
                logger.warning("Aspect '%s' failed for %s: %s",
                               aspect, content.metadata.paper_id, exc)
                fallback = {
                    "contribution": heuristic.core_contribution,
                    "methodology": heuristic.methodology,
                    "experiments": heuristic.experiments,
                    "critique": heuristic.critique,
                }
                aspects[aspect] = fallback[aspect]
        md = content.metadata
        return AnalysisReport(
            paper_id=md.paper_id,
            arxiv_id=md.arxiv_id,
            doi=md.doi,
            title=md.title,
            authors=list(md.authors),
            year=md.year,
            venue=md.venue,
            citation_count=md.citation_count,
            core_contribution=aspects["contribution"],  # type: ignore[arg-type]
            methodology=aspects["methodology"],  # type: ignore[arg-type]
            experiments=aspects["experiments"],  # type: ignore[arg-type]
            critique=aspects["critique"],  # type: ignore[arg-type]
            overall_assessment=(
                f"{aspects['contribution'].one_line_summary} "  # type: ignore[attr-defined]
                f"(confidence: aspects merged from chain mode)"
            ),
            confidence="medium",
            generated_by=f"llm:{model_tag}:chain",
        )


# --------------------------------------------------------------------------- #
# Module 4 — Synthesizer                                                      #
# --------------------------------------------------------------------------- #


def _bibtex_escape(value: str) -> str:
    """Escape LaTeX-special characters in BibTeX field values.

    Args:
        value: Raw field value.

    Returns:
        Escaped value safe for BibTeX.
    """
    return (
        value.replace("\\", "")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _bibtex_author(name: str) -> str:
    """Format one author name as ``Last, First``.

    Args:
        name: Display name ("First Last" or already "Last, First").

    Returns:
        BibTeX-formatted author.
    """
    name = _clean_ws(name)
    if not name:
        return "Anonymous"
    if "," in name:
        return name
    parts = name.split()
    if len(parts) == 1:
        return name
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _bibtex_key(report: AnalysisReport) -> str:
    """Build a citation key ``firstauthor+year+titleword``.

    Args:
        report: Analysis report carrying metadata.

    Returns:
        Sanitised citation key.
    """
    last = "anon"
    if report.authors:
        first = _clean_ws(report.authors[0])
        last = first.split(",")[0].split()[-1] if first else "anon"
    title_word = "paper"
    for word in re.split(r"[^A-Za-z0-9]+", report.title):
        if len(word) >= 4 and word.lower() not in {"with", "from", "towards",
                                                    "using", "based"}:
            title_word = word
            break
    key = f"{last}{report.year or 'nd'}{title_word}".lower()
    return re.sub(r"[^a-z0-9]", "", key) or "anonymous"


def to_bibtex(report: AnalysisReport) -> str:
    """Render one :class:`AnalysisReport` as a BibTeX entry.

    Uses ``@inproceedings`` for known conferences, ``@article`` for journals
    and ``@article`` with an arXiv journal string for preprints.

    Args:
        report: Analysis report with metadata snapshot.

    Returns:
        A complete BibTeX entry string.
    """
    authors = " and ".join(
        _bibtex_author(a) for a in report.authors
    ) or "Anonymous"
    key = _bibtex_key(report)
    venue = (report.venue or "").strip()
    norm_venue = _norm_venue(venue)
    is_journal = any(j in norm_venue for j in _JOURNAL_VENUES)
    lines: List[str]
    if venue and is_journal:
        lines = [
            f"@article{{{key},",
            f"  title = {{{_bibtex_escape(report.title)}}},",
            f"  author = {{{_bibtex_escape(authors)}}},",
            f"  journal = {{{_bibtex_escape(venue)}}},",
            f"  year = {{{report.year or 'n.d.'}}},",
        ]
    elif venue:
        lines = [
            f"@inproceedings{{{key},",
            f"  title = {{{_bibtex_escape(report.title)}}},",
            f"  author = {{{_bibtex_escape(authors)}}},",
            f"  booktitle = {{{_bibtex_escape(venue)}}},",
            f"  year = {{{report.year or 'n.d.'}}},",
        ]
    else:
        journal = (
            f"arXiv preprint arXiv:{report.arxiv_id}" if report.arxiv_id
            else "Preprint"
        )
        lines = [
            f"@article{{{key},",
            f"  title = {{{_bibtex_escape(report.title)}}},",
            f"  author = {{{_bibtex_escape(authors)}}},",
            f"  journal = {{{_bibtex_escape(journal)}}},",
            f"  year = {{{report.year or 'n.d.'}}},",
        ]
    if report.arxiv_id:
        lines.append(f"  eprint = {{{report.arxiv_id}}},")
        lines.append("  archivePrefix = {arXiv},")
    if report.doi:
        lines.append(f"  doi = {{{_bibtex_escape(report.doi)}}},")
    lines.append("}")
    return "\n".join(lines)


class Synthesizer:
    """Cross-paper synthesis: comparison matrix, BibTeX, literature review.

    Args:
        config: Skill configuration.
        llm: Optional LLM backend for review generation.
    """

    def __init__(
        self,
        config: MLPaperAnalystConfig,
        llm: Optional[LLMBackend] = None,
    ) -> None:
        self._config = config
        self._llm = llm

    # ------------------------------------------------------------------ #
    # Comparison matrix                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _md_cell(value: str, limit: int = 90) -> str:
        """Sanitise a value for a Markdown table cell.

        Args:
            value: Raw text.
            limit: Cell character limit.

        Returns:
            Safe, truncated cell text.
        """
        return _clean_ws(value).replace("|", "/")[:limit] or "-"

    def build_comparison_matrix(self, reports: List[AnalysisReport]) -> str:
        """Build the horizontal comparison Markdown matrix.

        Args:
            reports: Per-paper analyses.

        Returns:
            Markdown table string.
        """
        if not reports:
            return "_No papers to compare._"
        ordered = sorted(reports, key=lambda r: (r.year or 9999, r.title))
        header = (
            "| Paper | Year | Venue | Category | Core Contribution | "
            "Key Result / SOTA | Citations |"
        )
        separator = "|---|---|---|---|---|---|---|"
        rows = [header, separator]
        for r in ordered:
            title = r.title
            if r.arxiv_id:
                title = f"[{title}](https://arxiv.org/abs/{r.arxiv_id})"
            rows.append(
                "| {paper} | {year} | {venue} | {cat} | {core} | {sota} | {cites} |".format(
                    paper=self._md_cell(title, 70),
                    year=r.year or "-",
                    venue=self._md_cell(r.venue or "preprint", 24),
                    cat=self._md_cell(r.methodology.method_category, 30),
                    core=self._md_cell(
                        r.core_contribution.one_line_summary, 110
                    ),
                    sota=self._md_cell(
                        r.experiments.sota_improvement, 90
                    ),
                    cites=r.citation_count if r.citation_count is not None else "-",
                )
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # BibTeX                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_bibtex(reports: List[AnalysisReport]) -> List[str]:
        """Generate BibTeX entries for every analysed paper.

        Args:
            reports: Per-paper analyses.

        Returns:
            List of BibTeX entry strings.
        """
        return [to_bibtex(r) for r in reports]

    # ------------------------------------------------------------------ #
    # Literature review                                                  #
    # ------------------------------------------------------------------ #
    def _template_review(
        self,
        topic: str,
        reports: List[AnalysisReport],
        matrix_md: str,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """Assemble a structured review without an LLM.

        Args:
            topic: Review topic.
            reports: Per-paper analyses.
            matrix_md: Pre-built comparison matrix.

        Returns:
            Tuple ``(markdown, timeline, challenges, future_directions)``.
        """
        ordered = sorted(reports, key=lambda r: (r.year or 9999, r.title))
        timeline = [
            f"{r.year or 'n.d.'} · {r.title} — "
            f"{r.core_contribution.one_line_summary}"
            for r in ordered
        ]
        categories: Dict[str, List[AnalysisReport]] = {}
        for r in ordered:
            categories.setdefault(
                r.methodology.method_category, []
            ).append(r)

        lines: List[str] = [
            f"# Literature Review: {topic}",
            "",
            f"_Auto-generated by ml_paper_analyst from {len(reports)} papers._",
            "",
            "## 1. Overview & Scope",
            "",
            f"This survey covers {len(reports)} papers on **{topic}** "
            "retrieved from arXiv and Semantic Scholar, spanning "
            f"{ordered[0].year or 'n.d.'}–{ordered[-1].year or 'n.d.'}.",
            "",
            "## 2. Field Evolution Timeline",
            "",
        ]
        lines.extend(f"- {item}" for item in timeline)
        lines += ["", "## 3. Method Landscape", ""]
        for cat, group in categories.items():
            lines.append(f"### {cat} ({len(group)} papers)")
            lines.append("")
            for r in group:
                lines.append(
                    f"- **{r.title}** — {r.core_contribution.one_line_summary}"
                )
            lines.append("")
        lines += ["## 4. Comparative Analysis", "", matrix_md, ""]
        lines += ["## 5. Open Challenges", ""]
        challenges: List[str] = []
        seen: Set[str] = set()
        for r in ordered:
            for lim in r.critique.limitations:
                snippet = _clean_ws(lim)[:160]
                key = snippet.lower()
                if snippet and key not in seen:
                    seen.add(key)
                    challenges.append(snippet)
        lines.extend(f"- {c}" for c in challenges[:12] or ["- _None reported._"])
        lines += ["", "## 6. Future Directions", ""]
        futures: List[str] = []
        seen_f: Set[str] = set()
        for r in ordered:
            for direction in r.critique.future_directions:
                snippet = _clean_ws(direction)[:160]
                key = snippet.lower()
                if snippet and key not in seen_f:
                    seen_f.add(key)
                    futures.append(snippet)
        lines.extend(f"- {f}" for f in futures[:12] or ["- _None reported._"])
        lines += [
            "",
            "## 7. Conclusion",
            "",
            (
                f"The surveyed works trace a clear progression on {topic}: "
                + "; ".join(timeline[-3:])
                if len(timeline) >= 3 else
                f"The surveyed works address {topic} from complementary "
                "angles; see the comparative matrix above."
            ),
            "",
        ]
        return "\n".join(lines), timeline, challenges[:12], futures[:12]

    async def literature_review(
        self, topic: str, reports: List[AnalysisReport]
    ) -> LiteratureReview:
        """Generate the cross-paper literature review.

        Uses the LLM when configured, otherwise the deterministic template.

        Args:
            topic: Review topic / research intent.
            reports: Completed per-paper analyses.

        Returns:
            A complete :class:`LiteratureReview`.
        """
        matrix_md = self.build_comparison_matrix(reports)
        bibtex_entries = self.build_bibtex(reports)
        if not reports:
            return LiteratureReview(
                topic=topic,
                review_markdown="_No reports available to synthesise._",
                comparison_matrix_markdown=matrix_md,
                bibtex_entries=[],
                generated_by="template-fallback",
            )
        if self._llm is None:
            md_text, timeline, challenges, futures = self._template_review(
                topic, reports, matrix_md
            )
            return LiteratureReview(
                topic=topic,
                papers_analyzed=[r.title for r in reports],
                review_markdown=(
                    md_text + "\n\n## References\n\n```bibtex\n"
                    + "\n\n".join(bibtex_entries) + "\n```"
                ),
                comparison_matrix_markdown=matrix_md,
                bibtex_entries=bibtex_entries,
                evolution_timeline=timeline,
                open_challenges=challenges,
                future_directions=futures,
                generated_by="template-fallback",
            )

        digest = render_synthesis_digest(topic, reports)
        prompt = SYNTHESIS_USER_TEMPLATE.format(topic=topic, digest=digest)
        try:
            raw = await self._llm.complete(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=prompt,
                temperature=0.3,
                max_tokens=self._config.llm_max_tokens,
            )
            data = extract_json_block(raw)
            if data is None:
                raise RuntimeError("no JSON in synthesis response")
            review_md = str(data.get("review_markdown") or "").strip()
            if not review_md:
                raise RuntimeError("empty review_markdown")
            timeline = [str(x) for x in (data.get("evolution_timeline") or [])][:20]
            challenges = [str(x) for x in (data.get("open_challenges") or [])][:15]
            futures = [str(x) for x in (data.get("future_directions") or [])][:15]
            model_tag = getattr(self._llm, "model", "custom")
            return LiteratureReview(
                topic=topic,
                papers_analyzed=[r.title for r in reports],
                review_markdown=(
                    review_md
                    + "\n\n## Comparative Matrix\n\n" + matrix_md
                    + "\n\n## References\n\n```bibtex\n"
                    + "\n\n".join(bibtex_entries) + "\n```"
                ),
                comparison_matrix_markdown=matrix_md,
                bibtex_entries=bibtex_entries,
                evolution_timeline=timeline,
                open_challenges=challenges,
                future_directions=futures,
                generated_by=f"llm:{model_tag}",
            )
        except Exception as exc:  # noqa: BLE001 - fall back to template
            logger.warning("LLM synthesis failed (%s); using template", exc)
            md_text, timeline, challenges, futures = self._template_review(
                topic, reports, matrix_md
            )
            return LiteratureReview(
                topic=topic,
                papers_analyzed=[r.title for r in reports],
                review_markdown=(
                    md_text + "\n\n## References\n\n```bibtex\n"
                    + "\n\n".join(bibtex_entries) + "\n```"
                ),
                comparison_matrix_markdown=matrix_md,
                bibtex_entries=bibtex_entries,
                evolution_timeline=timeline,
                open_challenges=challenges,
                future_directions=futures,
                generated_by="template-fallback",
            )


# --------------------------------------------------------------------------- #
# Public module-level tool API (mirrors manifest.json schemas)                #
# --------------------------------------------------------------------------- #


async def search_ml_papers(
    query: str,
    max_results: int = 5,
    years: Optional[List[int]] = None,
    venue_filter: Optional[List[str]] = None,
    min_citations: int = 0,
    sources: Optional[List[str]] = None,
    config: Optional[MLPaperAnalystConfig] = None,
) -> List[PaperMetadata]:
    """Search ML papers across arXiv / Semantic Scholar / HF Daily Papers.

    Runs taxonomy-based query expansion, merges and deduplicates sources,
    enriches records via the Semantic Scholar batch API when filters need
    citation/venue data, then filters and ranks.

    Args:
        query: Natural-language research intent or keyword query.
        max_results: Maximum number of papers returned (default 5).
        years: Optional year filter; the list is interpreted as a
            min-max inclusive range (e.g. ``[2022, 2024]``).
        venue_filter: Optional venue names (aliases accepted, e.g.
            ``["NeurIPS", "CVPR"]``).
        min_citations: Optional citation threshold (requires enrichment).
        sources: Optional subset of ``["arxiv", "semantic_scholar",
            "huggingface_daily", "paperswithcode"]``.
        config: Optional configuration override.

    Returns:
        Ranked list of paper metadata records.

    Raises:
        ValueError: On invalid arguments.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    cfg = config or DEFAULT_CONFIG
    async with HTTPClient(cfg) as http:
        searcher = Searcher(
            cfg, http,
            cache=DiskCache(cfg) if cfg.cache_enabled else None,
            llm=None,
        )
        papers, _ = await searcher.search(
            intent=query.strip(),
            max_results=max_results,
            years=years,
            venue_filter=venue_filter,
            min_citations=min_citations,
            sources=sources,
            expand=True,
        )
    return papers


async def fetch_and_parse_paper(
    paper_id_or_url: str,
    config: Optional[MLPaperAnalystConfig] = None,
) -> PaperContent:
    """Fetch and parse one paper's full text into structured sections.

    Fallback chain: ar5iv HTML → arXiv native HTML → official PDF
    (PyMuPDF) → metadata-only.

    Args:
        paper_id_or_url: arXiv id / URL, DOI, or Semantic Scholar id.
        config: Optional configuration override.

    Returns:
        Structured :class:`PaperContent`.

    Raises:
        ValueError: When the identifier cannot be resolved.
    """
    cfg = config or DEFAULT_CONFIG
    async with HTTPClient(cfg) as http:
        extractor = Extractor(
            cfg, http, cache=DiskCache(cfg) if cfg.cache_enabled else None
        )
        return await extractor.fetch_and_parse(paper_id_or_url)


async def analyze_single_paper(
    paper_content: PaperContent,
    config: Optional[MLPaperAnalystConfig] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> AnalysisReport:
    """Run the four-aspect deep analysis on one parsed paper.

    Aspects: (1) Core Contribution & Novelty, (2) Technical Methodology,
    (3) Experimental Validation, (4) Critical Thinking.

    Args:
        paper_content: Parsed paper content from
            :func:`fetch_and_parse_paper`.
        config: Optional configuration override.
        llm_backend: Optional LLM backend (defaults to config-resolved
            backend; falls back to heuristic analysis when absent).

    Returns:
        Completed :class:`AnalysisReport`.
    """
    cfg = config or DEFAULT_CONFIG
    llm = resolve_llm_backend(cfg, llm_backend)
    analyzer = Analyzer(cfg, llm=llm)
    return await analyzer.analyze(paper_content)


async def synthesize_literature_review(
    topic: str,
    reports: List[AnalysisReport],
    config: Optional[MLPaperAnalystConfig] = None,
    llm_backend: Optional[LLMBackend] = None,
) -> LiteratureReview:
    """Synthesise analyses into a literature review with matrix + BibTeX.

    Note: returns a structured :class:`LiteratureReview`; the Markdown
    document is available as its ``review_markdown`` attribute.

    Args:
        topic: Review topic / original research intent.
        reports: Analysis reports to synthesise.
        config: Optional configuration override.
        llm_backend: Optional LLM backend override.

    Returns:
        Complete :class:`LiteratureReview`.
    """
    cfg = config or DEFAULT_CONFIG
    llm = resolve_llm_backend(cfg, llm_backend)
    synthesizer = Synthesizer(cfg, llm=llm)
    return await synthesizer.literature_review(topic, reports)
