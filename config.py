"""Configuration for the ``ml_paper_analyst`` skill.

All tunables (API credentials, rate limits, concurrency, caching, extraction
and LLM analysis behaviour) live here.  A ready-to-use :data:`DEFAULT_CONFIG`
instance is provided; override values via environment variables through
:meth:`MLPaperAnalystConfig.from_env` or by constructing the model directly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "MLPaperAnalystConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_RATE_LIMITS",
    "SUPPORTED_SOURCES",
]

#: Data sources understood by the searcher.
SUPPORTED_SOURCES: List[str] = [
    "arxiv",
    "semantic_scholar",
    "huggingface_daily",
    "paperswithcode",
]

#: Minimum interval (seconds) between consecutive requests, keyed by host.
#: arXiv asks for at most one request per 3 seconds; Semantic Scholar allows
#  ~1 rps unauthenticated (100 rps with a key, but we stay conservative).
DEFAULT_RATE_LIMITS: Dict[str, float] = {
    "export.arxiv.org": 3.2,
    "arxiv.org": 1.0,
    "ar5iv.labs.arxiv.org": 0.8,
    "api.semanticscholar.org": 1.1,
    "huggingface.co": 0.5,
    "paperswithcode.com": 0.5,
}


def _env_bool(raw: Optional[str], default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        raw: Raw environment variable value (``None`` when unset).
        default: Value to use when the variable is unset or empty.

    Returns:
        Parsed boolean.
    """
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(raw: Optional[str], default: float) -> float:
    """Parse a float environment variable with fallback.

    Args:
        raw: Raw environment variable value.
        default: Value to use when unset or unparsable.

    Returns:
        Parsed float.
    """
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(raw: Optional[str], default: int) -> int:
    """Parse an integer environment variable with fallback.

    Args:
        raw: Raw environment variable value.
        default: Value to use when unset or unparsable.

    Returns:
        Parsed integer.
    """
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


class MLPaperAnalystConfig(BaseModel):
    """Runtime configuration for the whole skill pipeline.

    Attributes mirror the constructor arguments documented below.  The model is
    frozen-friendly (``extra="forbid"``) so typos fail fast at construction.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------ #
    # Credentials                                                        #
    # ------------------------------------------------------------------ #
    semantic_scholar_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_json_mode: bool = False

    # ------------------------------------------------------------------ #
    # Search defaults                                                    #
    # ------------------------------------------------------------------ #
    default_sources: List[str] = Field(
        default_factory=lambda: ["arxiv", "semantic_scholar"]
    )
    arxiv_sort_by: str = "relevance"
    arxiv_categories: Optional[List[str]] = None
    #: Over-fetch factor: each source queries ``max_results * factor`` items
    #: before merging, filtering and ranking, so that filters do not starve
    #: the final result set.
    fetch_extra_factor: int = 3
    per_source_limit_cap: int = 100

    # ------------------------------------------------------------------ #
    # Network / concurrency / retry                                      #
    # ------------------------------------------------------------------ #
    request_timeout: float = 45.0
    max_concurrency: int = 6
    max_retries: int = 4
    backoff_base: float = 1.5
    backoff_max_delay: float = 30.0
    backoff_jitter: float = 0.25
    rate_limits: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_RATE_LIMITS)
    )
    default_rate_limit_interval: float = 0.5
    user_agent_contact: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Disk cache                                                         #
    # ------------------------------------------------------------------ #
    cache_enabled: bool = True
    cache_dir: Path = Path("~/.cache/ml_paper_analyst")
    cache_ttl: int = 7 * 24 * 3600
    content_cache_ttl: int = 30 * 24 * 3600

    # ------------------------------------------------------------------ #
    # Extractor                                                          #
    # ------------------------------------------------------------------ #
    html_sources: List[str] = Field(
        default_factory=lambda: ["ar5iv", "arxiv_html"]
    )
    pdf_fallback_enabled: bool = True
    max_section_chars: int = 30000
    max_figures: int = 15
    max_formulas: int = 80
    max_algorithms: int = 8
    max_pdf_chars: int = 200_000

    # ------------------------------------------------------------------ #
    # Analyzer / LLM                                                     #
    # ------------------------------------------------------------------ #
    analysis_mode: str = "single"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 3500
    analyzer_context_char_budget: int = 24000
    heuristic_fallback: bool = True

    # ------------------------------------------------------------------ #
    # Output                                                             #
    # ------------------------------------------------------------------ #
    output_dir: Path = Path("./mlpa_output")

    # ------------------------------------------------------------------ #
    # Validators                                                         #
    # ------------------------------------------------------------------ #
    @field_validator("request_timeout", "backoff_base", "backoff_max_delay",
                     "default_rate_limit_interval")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("max_concurrency", "max_retries", "max_section_chars",
                     "analyzer_context_char_budget", "per_source_limit_cap",
                     "fetch_extra_factor")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("analysis_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v not in {"single", "chain"}:
            raise ValueError('analysis_mode must be "single" or "chain"')
        return v

    @field_validator("arxiv_sort_by")
    @classmethod
    def _valid_sort(cls, v: str) -> str:
        if v not in {"relevance", "submittedDate", "lastSubmittedDate",
                     "updatedDate"}:
            raise ValueError("unsupported arXiv sortBy value")
        return v

    @field_validator("cache_dir", "output_dir")
    @classmethod
    def _expand_path(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("default_sources", "html_sources")
    @classmethod
    def _non_empty_list(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("must not be empty")
        return v

    # ------------------------------------------------------------------ #
    # Constructors / helpers                                             #
    # ------------------------------------------------------------------ #
    @property
    def user_agent(self) -> str:
        """Polite User-Agent string used for every outbound request."""
        contact = f"; mailto:{self.user_agent_contact}" if self.user_agent_contact else ""
        return f"ml-paper-analyst/1.0 (+https://github.com/ml-paper-analyst{contact})"

    @classmethod
    def from_env(cls, **overrides: Any) -> "MLPaperAnalystConfig":
        """Build a config from environment variables.

        Recognised variables:

        - ``SEMANTIC_SCHOLAR_API_KEY``
        - ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
        - ``MLPA_LLM_MODEL`` / ``MLPA_LLM_JSON_MODE``
        - ``MLPA_CACHE_DIR`` / ``MLPA_OUTPUT_DIR`` / ``MLPA_CACHE_ENABLED``
        - ``MLPA_MAX_CONCURRENCY`` / ``MLPA_REQUEST_TIMEOUT`` / ``MLPA_MAX_RETRIES``
        - ``MLPA_ANALYSIS_MODE`` / ``MLPA_LLM_TEMPERATURE``
        - ``MLPA_USER_AGENT_CONTACT``

        Args:
            **overrides: Explicit keyword overrides applied last (highest
                priority).

        Returns:
            A validated :class:`MLPaperAnalystConfig` instance.
        """
        data: Dict[str, Any] = {
            "semantic_scholar_api_key":
                os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
            "openai_api_key": os.environ.get("OPENAI_API_KEY") or None,
            "openai_base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "llm_model": os.environ.get("MLPA_LLM_MODEL", "gpt-4o-mini"),
            "llm_json_mode":
                _env_bool(os.environ.get("MLPA_LLM_JSON_MODE"), False),
            "cache_dir":
                Path(os.environ.get("MLPA_CACHE_DIR", "~/.cache/ml_paper_analyst")),
            "cache_enabled":
                _env_bool(os.environ.get("MLPA_CACHE_ENABLED"), True),
            "output_dir":
                Path(os.environ.get("MLPA_OUTPUT_DIR", "./mlpa_output")),
            "max_concurrency":
                _env_int(os.environ.get("MLPA_MAX_CONCURRENCY"), 6),
            "request_timeout":
                _env_float(os.environ.get("MLPA_REQUEST_TIMEOUT"), 45.0),
            "max_retries": _env_int(os.environ.get("MLPA_MAX_RETRIES"), 4),
            "analysis_mode":
                os.environ.get("MLPA_ANALYSIS_MODE", "single"),
            "llm_temperature":
                _env_float(os.environ.get("MLPA_LLM_TEMPERATURE"), 0.2),
            "user_agent_contact":
                os.environ.get("MLPA_USER_AGENT_CONTACT") or None,
        }
        data.update(overrides)
        return cls(**data)


#: Module-level default configuration instance.
DEFAULT_CONFIG = MLPaperAnalystConfig()
