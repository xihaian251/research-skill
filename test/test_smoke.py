"""Offline smoke tests for ml_paper_analyst.

These tests never touch the network: they validate schemas, pure parsing
helpers, venue matching, BibTeX generation, the heuristic analyzer and the
JSON extractor.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_paper_analyst.config import MLPaperAnalystConfig  # noqa: E402
from ml_paper_analyst.schemas import (  # noqa: E402
    AnalysisReport,
    PaperContent,
    PaperMetadata,
)
from ml_paper_analyst.tools import (  # noqa: E402
    Analyzer,
    Extractor,
    Searcher,
    Synthesizer,
    extract_json_block,
    to_bibtex,
    venue_matches,
)
from ml_paper_analyst.prompts import render_analysis_context  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def config() -> MLPaperAnalystConfig:
    """Config with caching disabled and tmp paths."""
    return MLPaperAnalystConfig(
        cache_enabled=False,
        cache_dir=Path("./.test_cache"),
        output_dir=Path("./.test_output"),
    )


@pytest.fixture
def paper_md() -> PaperMetadata:
    """Synthetic paper metadata."""
    return PaperMetadata(
        paper_id="2401.00001",
        source="arxiv",
        title="LatentSplat: Text-to-3D via Score Distillation",
        authors=["Alice Zhang", "Bob Li", "Carol-Ann Smith"],
        abstract=(
            "Text-to-3D generation is slow. However, per-view score "
            "distillation is expensive. We propose LatentSplat, which "
            "distills 2D diffusion priors into 3D Gaussians and outperforms "
            "DreamFusion by 12% in CLIP score on ShapeNet and COCO."
        ),
        arxiv_id="2401.00001",
        year=2024,
        venue=None,
        pdf_url="https://arxiv.org/pdf/2401.00001",
        abs_url="https://arxiv.org/abs/2401.00001",
        comment="Accepted to CVPR 2024 as highlight",
    )


@pytest.fixture
def content(paper_md: PaperMetadata) -> PaperContent:
    """Synthetic parsed paper content."""
    return PaperContent(
        metadata=paper_md,
        sections={
            "abstract": paper_md.abstract,
            "introduction": "Intro text. " * 10,
            "method": "We use a diffusion U-Net backbone with SDS loss. " * 5,
            "experiments": (
                "We evaluate on ShapeNet and COCO with FID and CLIP score. "
                "Compared with DreamFusion and Magic3D baselines, we improve "
                "CLIP score by 12%."
            ),
            "limitations": (
                "The method requires 8 A100 GPUs. Generalisation to "
                "out-of-distribution prompts is weak."
            ),
            "conclusion": "We presented LatentSplat. Future work includes "
                          "video generation and faster sampling.",
        },
        figures=[],
        algorithms=[],
        formulas=[],
        extraction_source="ar5iv",
    )


def make_report(md: PaperMetadata, **overrides) -> AnalysisReport:
    """Build an AnalysisReport snapshot from metadata."""
    data = dict(
        paper_id=md.paper_id,
        arxiv_id=md.arxiv_id,
        doi=md.doi,
        title=md.title,
        authors=md.authors,
        year=md.year,
        venue=md.venue,
        citation_count=md.citation_count,
    )
    data.update(overrides)
    return AnalysisReport(**data)


# --------------------------------------------------------------------------- #
# Schema / util tests                                                         #
# --------------------------------------------------------------------------- #
def test_paper_metadata_dedupe_and_merge(paper_md: PaperMetadata) -> None:
    """Dedupe keys are stable and merge fills gaps."""
    s2_copy = paper_md.model_dump()
    s2_copy.update({
        "source": "semantic_scholar",
        "citation_count": 120,
        "venue": "CVPR",
    })
    merged = paper_md.merged_with(PaperMetadata.model_validate(s2_copy))
    assert merged.citation_count == 120
    assert merged.venue == "CVPR"
    assert merged.dedupe_key() == "arxiv:2401.00001"


def test_venue_matching_aliases(paper_md: PaperMetadata) -> None:
    """Venue matching tolerates aliases and arXiv comments."""
    assert venue_matches(paper_md, ["CVPR"]) is True
    assert venue_matches(paper_md, ["Conference on Computer Vision and "
                                    "Pattern Recognition"]) is True
    assert venue_matches(paper_md, ["NeurIPS"]) is False
    assert venue_matches(paper_md, []) is True


def test_section_canonicalisation() -> None:
    """Numbered/synonym headings map to canonical names."""
    assert Extractor.parse_arxiv_id("https://arxiv.org/abs/2312.06621v2") == "2312.06621"
    assert Extractor.parse_arxiv_id("2401.12345v3") == "2401.12345"
    assert Extractor.parse_arxiv_id("not-an-id") is None


def test_json_extractor_variants() -> None:
    """JSON extraction handles fences, prose and nesting."""
    assert extract_json_block('{"a": 1}') == {"a": 1}
    assert extract_json_block('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json_block('Sure! Here it is:\n{"a": {"b": "}"}} done') == {
        "a": {"b": "}"}
    }
    assert extract_json_block("no json here") is None


def test_query_expansion_taxonomy() -> None:
    """Taxonomy expansion adds the expected technical terms."""
    terms = Searcher._taxonomy_expand("diffusion text to 3d generation")
    joined = " ".join(terms).lower()
    assert "score distillation sampling" in joined
    assert "gaussian splatting" in joined
    assert len(terms) <= 10


# --------------------------------------------------------------------------- #
# Analyzer (heuristic mode)                                                   #
# --------------------------------------------------------------------------- #
def test_heuristic_analyzer(config, content) -> None:
    """Heuristic analyzer extracts datasets/metrics/baselines/limitations."""
    analyzer = Analyzer(config, llm=None)
    report = asyncio.run(analyzer.analyze(content))
    assert report.generated_by == "heuristic-fallback"
    assert report.confidence == "low"
    assert "ShapeNet" in report.experiments.datasets
    assert any("CLIP" in m for m in report.experiments.metrics)
    assert "DreamFusion" in report.experiments.baselines
    assert report.experiments.sota_improvement  # found the 12% sentence
    assert report.critique.limitations
    assert report.methodology.method_category != "Uncategorized"
    assert report.core_contribution.one_line_summary


def test_render_analysis_context_budget(content) -> None:
    """Context renderer respects the character budget."""
    text = render_analysis_context(content, 4000)
    assert "TITLE:" in text
    assert "SECTION: METHOD" in text
    assert len(text) < 12000


# --------------------------------------------------------------------------- #
# Synthesizer                                                                 #
# --------------------------------------------------------------------------- #
def test_bibtex_generation(paper_md: PaperMetadata) -> None:
    """BibTeX picks @inproceedings for venues, arXiv journal otherwise."""
    with_venue = make_report(paper_md, venue="CVPR")
    entry = to_bibtex(with_venue)
    assert entry.startswith("@inproceedings{")
    assert "booktitle = {CVPR}" in entry
    assert "eprint = {2401.00001}" in entry

    preprint = make_report(paper_md, venue=None)
    entry2 = to_bibtex(preprint)
    assert entry2.startswith("@article{")
    assert "arXiv preprint arXiv:2401.00001" in entry2
    assert "Zhang" in entry2 and "Li" in entry2  # author formatting


def test_synthesizer_template_fallback(config, paper_md) -> None:
    """Template review assembles matrix/timeline/bibtex without an LLM."""
    report = make_report(paper_md)
    synth = Synthesizer(config, llm=None)
    review = asyncio.run(synth.literature_review("text-to-3d", [report]))
    assert review.generated_by == "template-fallback"
    assert review.review_markdown.startswith("# Literature Review")
    assert "| Paper |" in review.comparison_matrix_markdown
    assert review.bibtex_entries
    assert review.evolution_timeline


# --------------------------------------------------------------------------- #
# Config / pipeline plumbing                                                  #
# --------------------------------------------------------------------------- #
def test_config_from_env(monkeypatch) -> None:
    """from_env parses environment overrides."""
    monkeypatch.setenv("MLPA_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("MLPA_ANALYSIS_MODE", "chain")
    monkeypatch.setenv("MLPA_CACHE_ENABLED", "false")
    cfg = MLPaperAnalystConfig.from_env()
    assert cfg.max_concurrency == 3
    assert cfg.analysis_mode == "chain"
    assert cfg.cache_enabled is False


def test_config_validation() -> None:
    """Invalid values fail fast."""
    with pytest.raises(Exception):
        MLPaperAnalystConfig(max_concurrency=0)
    with pytest.raises(Exception):
        MLPaperAnalystConfig(analysis_mode="bogus")
