"""Skill entry point — the one-click literature-analysis pipeline.

:class:`MLPaperAnalystSkill` wires the four capability modules (Searcher →
Extractor → Analyzer → Synthesizer) into a single resumable pipeline run,
sharing one HTTP session, one disk cache and one LLM backend across all
stages, and persists the resulting artifacts to disk.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MLPaperAnalystConfig
from .schemas import (
    AnalysisReport,
    LiteratureReview,
    PaperContent,
    PaperMetadata,
    SearchQuery,
    SkillRunResult,
)
from .tools import (
    Analyzer,
    DiskCache,
    Extractor,
    HTTPClient,
    LLMBackend,
    Searcher,
    Synthesizer,
    resolve_llm_backend,
)

logger = logging.getLogger(__name__)

__all__ = ["MLPaperAnalystSkill"]


def _slugify_topic(topic: str) -> str:
    """Convert a topic into a filesystem-safe artifact basename.

    Args:
        topic: Raw research topic.

    Returns:
        Lowercase slug limited to 60 characters.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")
    return slug[:60] or "report"


class MLPaperAnalystSkill:
    """End-to-end ML literature analysis skill.

    Example::

        skill = MLPaperAnalystSkill()   # reads env vars
        result = skill.run_sync(
            topic="diffusion models for text-to-3D",
            max_results=8,
            years=[2022, 2026],
            venue_filter=["CVPR", "ICLR", "NeurIPS"],
        )
        print(result.review.review_markdown)

    Args:
        config: Optional configuration override (defaults to
            :meth:`MLPaperAnalystConfig.from_env`).
        llm_backend: Optional injected LLM backend; when ``None`` the
            backend is resolved from config (OpenAI-compatible API key),
            and the skill degrades to heuristic analysis without one.
    """

    def __init__(
        self,
        config: Optional[MLPaperAnalystConfig] = None,
        llm_backend: Optional[LLMBackend] = None,
    ) -> None:
        self.config = config or MLPaperAnalystConfig.from_env()
        self._llm = resolve_llm_backend(self.config, llm_backend)
        if self._llm is None:
            logger.info(
                "No LLM backend configured — analysis/synthesis will use "
                "heuristic/template fallbacks."
            )

    # ------------------------------------------------------------------ #
    # Pipeline                                                           #
    # ------------------------------------------------------------------ #
    async def run(
        self,
        topic: str,
        max_results: int = 10,
        years: Optional[List[int]] = None,
        venue_filter: Optional[List[str]] = None,
        min_citations: int = 0,
        sources: Optional[List[str]] = None,
        expand_query: bool = True,
        fetch_full_text: bool = True,
        analysis_mode: Optional[str] = None,
        output_dir: Optional[Path] = None,
        save_artifacts: bool = True,
    ) -> SkillRunResult:
        """Execute the full pipeline and persist artifacts.

        Stages: search → fetch/parse → analyse (per paper, concurrent) →
        synthesise → export (Markdown review, comparison matrix, BibTeX,
        full JSON dump).

        Args:
            topic: Natural-language research intent.
            max_results: Number of papers to analyse.
            years: Optional year range (list interpreted as min–max).
            venue_filter: Optional venue restriction (aliases accepted).
            min_citations: Optional citation threshold.
            sources: Optional data-source subset.
            expand_query: Whether to expand the query (LLM/taxonomy).
            fetch_full_text: When False, skip full-text fetching and
                analyse abstracts only (much faster).
            analysis_mode: ``"single"`` or ``"chain"`` override.
            output_dir: Directory for artifacts (defaults to config).
            save_artifacts: Whether to write artifact files.

        Returns:
            A :class:`SkillRunResult` with everything the run produced.

        Raises:
            ValueError: When ``topic`` is empty or arguments are invalid.
        """
        if not topic or not topic.strip():
            raise ValueError("topic must be a non-empty string")
        topic = topic.strip()
        stats: Dict[str, Any] = {"started_at": time.time()}
        warnings: List[str] = []
        result = SkillRunResult(topic=topic, warnings=warnings, stats=stats)

        cfg = self.config
        cache = DiskCache(cfg) if cfg.cache_enabled else None
        semaphore = asyncio.Semaphore(cfg.max_concurrency)

        async with HTTPClient(cfg) as http:
            # ---- Stage 1: search ---------------------------------------- #
            t0 = time.perf_counter()
            searcher = Searcher(cfg, http, cache=cache, llm=self._llm)
            papers, query = await searcher.search(
                intent=topic,
                max_results=max_results,
                years=years,
                venue_filter=venue_filter,
                min_citations=min_citations,
                sources=sources,
                expand=expand_query,
            )
            stats["search_seconds"] = round(time.perf_counter() - t0, 2)
            stats["papers_found"] = len(papers)
            result.papers = papers
            result.query = query
            if not papers:
                warnings.append("no papers matched the query/filters")
                logger.warning("Search returned 0 papers for %r", topic)
                if save_artifacts:
                    result.artifacts = self._save_artifacts(
                        result, output_dir
                    )
                return result

            # ---- Stage 2: fetch & parse full text ----------------------- #
            t0 = time.perf_counter()
            extractor = Extractor(cfg, http, cache=cache)

            async def _fetch(paper: PaperMetadata) -> PaperContent:
                if not fetch_full_text:
                    return PaperContent(
                        metadata=paper,
                        sections={"abstract": paper.abstract}
                        if paper.abstract else {},
                        extraction_source="metadata_only",
                        parse_warnings=["full-text fetch disabled by caller"],
                    )
                async with semaphore:
                    return await extractor.fetch_and_parse(paper)

            contents = await asyncio.gather(
                *(_fetch(p) for p in papers), return_exceptions=True
            )
            parsed: List[PaperContent] = []
            for paper, outcome in zip(papers, contents):
                if isinstance(outcome, BaseException):
                    warnings.append(
                        f"fetch failed for {paper.paper_id}: {outcome}"
                    )
                    parsed.append(PaperContent(
                        metadata=paper,
                        sections={"abstract": paper.abstract}
                        if paper.abstract else {},
                        parse_warnings=[f"fetch error: {outcome}"],
                    ))
                else:
                    parsed.append(outcome)
            result.contents = parsed
            fulltext_count = sum(1 for c in parsed if c.has_fulltext)
            stats["fetch_seconds"] = round(time.perf_counter() - t0, 2)
            stats["fulltext_extracted"] = fulltext_count

            # ---- Stage 3: per-paper analysis ---------------------------- #
            t0 = time.perf_counter()
            analyzer = Analyzer(cfg, llm=self._llm, analysis_mode=analysis_mode)

            async def _analyze(content: PaperContent) -> AnalysisReport:
                async with semaphore:
                    try:
                        return await analyzer.analyze(content)
                    except Exception as exc:  # noqa: BLE001 - keep pipeline
                        warnings.append(
                            f"analysis failed for "
                            f"{content.metadata.paper_id}: {exc}"
                        )
                        return analyzer._heuristic_report(content)

            result.reports = list(await asyncio.gather(
                *(_analyze(c) for c in parsed)
            ))
            stats["analysis_seconds"] = round(time.perf_counter() - t0, 2)
            llm_reports = sum(
                1 for r in result.reports if r.generated_by.startswith("llm")
            )
            stats["llm_analyses"] = llm_reports
            stats["heuristic_analyses"] = len(result.reports) - llm_reports

            # ---- Stage 4: synthesis ------------------------------------- #
            t0 = time.perf_counter()
            if result.reports:
                synthesizer = Synthesizer(cfg, llm=self._llm)
                result.review = await synthesizer.literature_review(
                    topic, result.reports
                )
            else:
                warnings.append("no analyses produced; skipped synthesis")
            stats["synthesis_seconds"] = round(time.perf_counter() - t0, 2)

        stats["total_seconds"] = round(time.time() - stats["started_at"], 2)
        stats["warnings_count"] = len(warnings)
        if save_artifacts:
            result.artifacts.update(self._save_artifacts(result, output_dir))
        return result

    def run_sync(self, **kwargs: Any) -> SkillRunResult:
        """Synchronous wrapper around :meth:`run`.

        Args:
            **kwargs: Same arguments as :meth:`run`.

        Returns:
            The pipeline result.

        Raises:
            RuntimeError: When called from inside a running event loop
                (use ``await run(...)`` there instead).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(**kwargs))
        raise RuntimeError(
            "run_sync() cannot be called inside a running event loop; "
            "await run(...) instead"
        )

    # ------------------------------------------------------------------ #
    # Artifact persistence                                               #
    # ------------------------------------------------------------------ #
    def _save_artifacts(
        self,
        result: SkillRunResult,
        output_dir: Optional[Path],
    ) -> Dict[str, str]:
        """Write the review / matrix / BibTeX / JSON artifacts to disk.

        Args:
            result: Completed pipeline result (mutated with artifact paths).
            output_dir: Target directory override.

        Returns:
            Mapping of artifact name → written file path.
        """
        base = (output_dir or self.config.output_dir).expanduser()
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            result.warnings.append(f"cannot create output dir {base}: {exc}")
            return {}
        slug = _slugify_topic(result.topic)
        artifacts: Dict[str, str] = {}
        review: Optional[LiteratureReview] = result.review

        def _write(name: str, suffix: str, text: str) -> None:
            path = base / f"{slug}{suffix}"
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                artifacts[name] = str(path)
            except OSError as exc:
                result.warnings.append(f"cannot write {path}: {exc}")

        if review is not None:
            _write("review_markdown", "_literature_review.md",
                   review.review_markdown or "")
            _write("comparison_matrix", "_comparison_matrix.md",
                   review.comparison_matrix_markdown or "")
            if review.bibtex_entries:
                _write("bibtex", "_references.bib",
                       "\n\n".join(review.bibtex_entries))
        _write(
            "full_result_json", "_full_result.json",
            json.dumps(
                result.model_dump(mode="json"), ensure_ascii=False, indent=2
            ),
        )
        logger.info("Artifacts written to %s: %s", base, list(artifacts))
        return artifacts

    # ------------------------------------------------------------------ #
    # Agent integration helpers                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def tool_manifest() -> Dict[str, Any]:
        """Return the Agent tool-calling schema declared in ``manifest.json``.

        Returns:
            Parsed manifest dictionary for dynamic agent registration.
        """
        manifest_path = Path(__file__).resolve().parent / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
