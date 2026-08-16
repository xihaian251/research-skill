"""Pydantic data models for the ``ml_paper_analyst`` skill.

The models below flow through the pipeline in this order::

    SearchQuery -> List[PaperMetadata] -> PaperContent -> AnalysisReport
                -> LiteratureReview / SkillRunResult
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SearchSource = Literal[
    "arxiv", "semantic_scholar", "huggingface_daily", "paperswithcode", "manual"
]
ExtractionSource = Literal[
    "ar5iv", "arxiv_html", "pdf", "metadata_only"
]


def _utcnow() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        Timezone-aware current UTC time.
    """
    return datetime.now(timezone.utc)


class PaperMetadata(BaseModel):
    """Bibliographic metadata for a single paper.

    Attributes:
        paper_id: Internal unique id (arXiv id, S2 id, DOI or slug).
        source: Which data source produced the record.
        title: Paper title.
        authors: Author names in display order.
        abstract: Paper abstract (may be empty for some sources).
        arxiv_id: Canonical arXiv identifier without version, if any.
        doi: Digital Object Identifier, if any.
        year: Publication year.
        published_date: ISO-8601 publication timestamp when known.
        venue: Publication venue (conference or journal), when known.
        citation_count: Total citations (Semantic Scholar).
        influential_citation_count: Influential citations (Semantic Scholar).
        fields_of_study: S2 field-of-study tags.
        tldr: Short one-sentence summary from S2 TLDR, when available.
        pdf_url: Direct PDF link.
        abs_url: Landing/abstract page link.
        html_url: ar5iv / arXiv HTML link (filled for arXiv papers).
        open_access_pdf_url: Open-access PDF link from S2, when available.
        comment: Raw arXiv comment field (often contains venue info).
        keywords: Extra keywords (e.g. from query expansion provenance).
    """

    paper_id: str
    source: SearchSource = "manual"
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    published_date: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    fields_of_study: List[str] = Field(default_factory=list)
    tldr: Optional[str] = None
    pdf_url: Optional[str] = None
    abs_url: Optional[str] = None
    html_url: Optional[str] = None
    open_access_pdf_url: Optional[str] = None
    comment: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)

    def dedupe_key(self) -> str:
        """Return a stable key used for cross-source deduplication.

        Returns:
            arXiv id if present, else DOI, else normalised title.
        """
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        normalised = "".join(ch for ch in self.title.lower() if ch.isalnum())
        return f"title:{normalised}"

    def merged_with(self, other: "PaperMetadata") -> "PaperMetadata":
        """Create a new record filling missing fields from ``other``.

        Args:
            other: Record to source missing values from.

        Returns:
            A new :class:`PaperMetadata` with fields of ``self`` taking
            priority.
        """
        data = self.model_dump()
        other_data = other.model_dump()
        for key, value in other_data.items():
            if data.get(key) in (None, "", []) and value not in (None, "", []):
                data[key] = value
        if not data.get("authors"):
            data["authors"] = other_data.get("authors") or []
        return PaperMetadata.model_validate(data)


class SearchQuery(BaseModel):
    """Provenance record of how a user intent became an actual query.

    Attributes:
        raw_intent: Original natural-language research intent.
        expanded_terms: Additional terms added by query expansion.
        final_query: Final disjunctive query string sent to sources.
        years: Year filter applied.
        venue_filter: Venue filter applied.
        min_citations: Citation threshold applied.
        sources: Data sources queried.
        expansion_method: ``taxonomy`` or ``llm`` (or ``none``).
    """

    raw_intent: str
    expanded_terms: List[str] = Field(default_factory=list)
    final_query: str = ""
    years: Optional[List[int]] = None
    venue_filter: Optional[List[str]] = None
    min_citations: int = 0
    sources: List[str] = Field(default_factory=list)
    expansion_method: str = "taxonomy"


class FigureInfo(BaseModel):
    """A figure or table caption extracted from the paper.

    Attributes:
        figure_id: Identifier such as ``Figure 3`` or ``Table 2``.
        caption: Full caption text.
        kind: ``figure`` or ``table``.
    """

    figure_id: str
    caption: str
    kind: str = "figure"


class AlgorithmInfo(BaseModel):
    """Pseudo-code / algorithm block extracted from the paper.

    Attributes:
        algorithm_id: Identifier such as ``Algorithm 1``.
        caption: Caption line (name of the algorithm).
        pseudocode: Plain-text pseudocode, when renderable.
    """

    algorithm_id: str
    caption: str = ""
    pseudocode: str = ""


class FormulaInfo(BaseModel):
    """A display equation extracted from the paper.

    Attributes:
        latex: LaTeX source recovered from the HTML ``alttext`` attribute.
        context: Surrounding paragraph snippet for interpretation.
    """

    latex: str
    context: str = ""


class PaperContent(BaseModel):
    """Structured full-text content of a single paper.

    Attributes:
        metadata: Bibliographic metadata of the paper.
        sections: Mapping of canonical section name (e.g. ``method``) to text.
        extra_sections: Named sections that had no canonical mapping.
        affiliations: Author affiliations, when recoverable from HTML.
        figures: Figure/table captions.
        algorithms: Algorithm pseudo-code blocks.
        formulas: Display equations with LaTeX and context.
        full_text: Raw concatenated text (PDF extraction path only).
        extraction_source: Which extraction strategy succeeded.
        parse_warnings: Non-fatal issues encountered while parsing.
    """

    metadata: PaperMetadata
    sections: Dict[str, str] = Field(default_factory=dict)
    extra_sections: Dict[str, str] = Field(default_factory=dict)
    affiliations: List[str] = Field(default_factory=list)
    figures: List[FigureInfo] = Field(default_factory=list)
    algorithms: List[AlgorithmInfo] = Field(default_factory=list)
    formulas: List[FormulaInfo] = Field(default_factory=list)
    full_text: Optional[str] = None
    extraction_source: ExtractionSource = "metadata_only"
    parse_warnings: List[str] = Field(default_factory=list)

    def section_text(self, *names: str, default: str = "") -> str:
        """Return concatenated text of the first existing section.

        Args:
            *names: Canonical section names in priority order.
            default: Value returned when none of the sections exist.

        Returns:
            Section text or ``default``.
        """
        for name in names:
            text = self.sections.get(name)
            if text:
                return text
        return default

    @property
    def has_fulltext(self) -> bool:
        """Whether any body section beyond the abstract was extracted."""
        return bool(
            any(k != "abstract" and v for k, v in self.sections.items())
            or self.extra_sections
        )


class CoreContribution(BaseModel):
    """Aspect 1: what the paper contributes and why it matters.

    Attributes:
        one_line_summary: Single-sentence statement of the contribution.
        motivation: Problem context and why prior work falls short.
        novelty: What is genuinely new versus the closest prior art.
        pain_points_addressed: Concrete pain points of baselines solved here.
    """

    one_line_summary: str = ""
    motivation: str = ""
    novelty: str = ""
    pain_points_addressed: List[str] = Field(default_factory=list)


class MethodologyBreakdown(BaseModel):
    """Aspect 2: technical methodology.

    Attributes:
        method_category: Coarse family (e.g. ``Diffusion Generative``).
        architecture: Model/system architecture description.
        loss_functions: Losses / training objectives.
        training_techniques: Optimisation, scheduling, data tricks.
        key_assumptions: Assumptions the method silently relies on.
    """

    method_category: str = "Uncategorized"
    architecture: str = ""
    loss_functions: str = ""
    training_techniques: str = ""
    key_assumptions: List[str] = Field(default_factory=list)


class ExperimentalValidation(BaseModel):
    """Aspect 3: experimental evidence.

    Attributes:
        datasets: Benchmarks / datasets used.
        metrics: Evaluation metrics reported.
        baselines: Methods compared against.
        sota_improvement: Quantified improvement over prior SOTA (text).
        main_results_summary: Narrative summary of the key results.
    """

    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    baselines: List[str] = Field(default_factory=list)
    sota_improvement: str = ""
    main_results_summary: str = ""


class Critique(BaseModel):
    """Aspect 4: critical assessment.

    Attributes:
        strengths: Main strengths of the work.
        limitations: Limitations (compute cost, generalisation, weak eval...).
        future_directions: Concrete follow-up research directions.
    """

    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Complete four-aspect analysis of a single paper.

    Attributes:
        paper_id / arxiv_id / doi / title / authors / year / venue /
        citation_count: Snapshot of metadata for downstream synthesis.
        core_contribution: Aspect 1 result.
        methodology: Aspect 2 result.
        experiments: Aspect 3 result.
        critique: Aspect 4 result.
        overall_assessment: One-paragraph verdict.
        confidence: Analyst confidence in the analysis.
        generated_by: ``llm:<model>`` or ``heuristic-fallback``.
        analyzed_at: UTC timestamp of analysis.
    """

    paper_id: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    core_contribution: CoreContribution = Field(default_factory=CoreContribution)
    methodology: MethodologyBreakdown = Field(
        default_factory=MethodologyBreakdown
    )
    experiments: ExperimentalValidation = Field(
        default_factory=ExperimentalValidation
    )
    critique: Critique = Field(default_factory=Critique)
    overall_assessment: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    generated_by: str = "heuristic-fallback"
    analyzed_at: datetime = Field(default_factory=_utcnow)


class LiteratureReview(BaseModel):
    """Cross-paper synthesis output.

    Attributes:
        topic: Review topic / research intent.
        generated_at: UTC timestamp.
        papers_analyzed: Titles of the papers included.
        review_markdown: Full literature-review report in Markdown.
        comparison_matrix_markdown: Horizontal comparison matrix in Markdown.
        bibtex_entries: One BibTeX string per paper.
        evolution_timeline: Ordered one-liners tracing the field's evolution.
        open_challenges: Aggregated open problems.
        future_directions: Aggregated future research directions.
        generated_by: ``llm:<model>`` or ``template-fallback``.
    """

    topic: str
    generated_at: datetime = Field(default_factory=_utcnow)
    papers_analyzed: List[str] = Field(default_factory=list)
    review_markdown: str = ""
    comparison_matrix_markdown: str = ""
    bibtex_entries: List[str] = Field(default_factory=list)
    evolution_timeline: List[str] = Field(default_factory=list)
    open_challenges: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)
    generated_by: str = "template-fallback"


class SkillRunResult(BaseModel):
    """Result of one end-to-end :meth:`skill.MLPaperAnalystSkill.run` call.

    Attributes:
        topic: Research topic passed in.
        query: Resolved search query provenance.
        papers: Final filtered/ranked paper set.
        contents: Parsed contents aligned with ``papers``.
        reports: Per-paper analysis reports aligned with ``papers``.
        review: Cross-paper synthesis (``None`` when no paper succeeded).
        artifacts: Mapping of artifact name to written file path.
        warnings: Non-fatal warnings accumulated during the run.
        stats: Per-stage timings and counters.
    """

    topic: str
    query: Optional[SearchQuery] = None
    papers: List[PaperMetadata] = Field(default_factory=list)
    contents: List[PaperContent] = Field(default_factory=list)
    reports: List[AnalysisReport] = Field(default_factory=list)
    review: Optional[LiteratureReview] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
