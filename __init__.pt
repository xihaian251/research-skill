"""ml_paper_analyst — production-grade ML literature-analysis Agent Skill.

Pipeline: cross-source search (arXiv / Semantic Scholar / HF Daily Papers)
→ full-text extraction (ar5iv / arXiv HTML / PDF) → four-aspect deep
analysis → cross-paper synthesis (matrix + BibTeX + literature review).

Quickstart::

    from ml_paper_analyst import MLPaperAnalystSkill

    skill = MLPaperAnalystSkill()
    result = skill.run_sync(topic="diffusion text to 3d", max_results=5)
    print(result.review.review_markdown)
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    # skill entry
    "MLPaperAnalystSkill",
    # config
    "MLPaperAnalystConfig",
    "DEFAULT_CONFIG",
    # schemas
    "PaperMetadata",
    "PaperContent",
    "FigureInfo",
    "AlgorithmInfo",
    "FormulaInfo",
    "SearchQuery",
    "CoreContribution",
    "MethodologyBreakdown",
    "ExperimentalValidation",
    "Critique",
    "AnalysisReport",
    "LiteratureReview",
    "SkillRunResult",
    # tools
    "search_ml_papers",
    "fetch_and_parse_paper",
    "analyze_single_paper",
    "synthesize_literature_review",
    "OpenAIChatBackend",
    "LLMBackend",
    "Searcher",
    "Extractor",
    "Analyzer",
    "Synthesizer",
]

from .config import DEFAULT_CONFIG, MLPaperAnalystConfig
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
    SkillRunResult,
)
from .tools import (
    Analyzer,
    Extractor,
    HTTPClient,  # noqa: F401  (re-exported for power users)
    LLMBackend,
    OpenAIChatBackend,
    Searcher,
    Synthesizer,
    analyze_single_paper,
    fetch_and_parse_paper,
    search_ml_papers,
    synthesize_literature_review,
)
from .skill import MLPaperAnalystSkill
