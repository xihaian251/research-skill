"""Prompt templates and renderers for the ``ml_paper_analyst`` skill.

All LLM interactions (query expansion, four-aspect paper analysis, literature
synthesis) are defined here so prompts can be tuned independently of code.

Every prompt enforces *strict JSON output* and *grounding in the provided
paper text* — the two properties the parsing logic in :mod:`tools` relies on.
"""
from __future__ import annotations

from typing import List

from .schemas import AnalysisReport, PaperContent

__all__ = [
    "QUERY_EXPANSION_SYSTEM_PROMPT",
    "QUERY_EXPANSION_USER_TEMPLATE",
    "ACADEMIC_ANALYST_SYSTEM_PROMPT",
    "CONTRIBUTION_USER_TEMPLATE",
    "METHODOLOGY_USER_TEMPLATE",
    "EXPERIMENTS_USER_TEMPLATE",
    "CRITIQUE_USER_TEMPLATE",
    "SINGLE_PASS_USER_TEMPLATE",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_USER_TEMPLATE",
    "ASPECT_JSON_SCHEMAS",
    "render_analysis_context",
    "render_synthesis_digest",
]

# --------------------------------------------------------------------------- #
# Module 1 — query expansion                                                  #
# --------------------------------------------------------------------------- #
QUERY_EXPANSION_SYSTEM_PROMPT = """\
You are an expert machine-learning research librarian.
Given a researcher's natural-language intent, produce a list of additional
search terms that will maximise recall on arXiv / Semantic Scholar.

Rules:
1. Include canonical technique names (e.g. "score distillation sampling"),
   landmark model names (e.g. "DreamFusion"), and standard paraphrases
   (e.g. "text-to-3D generation", "3D content creation").
2. Include 1-2 broader umbrella terms and 1-2 narrower sub-problem terms.
3. Return 6-12 terms total. Never repeat the original intent verbatim.
4. Lowercase everything except proper nouns/model names.
5. Respond with STRICT JSON only — no markdown fences, no commentary:
{"expanded_terms": ["term1", "term2", ...], "rationale": "one short sentence"}
"""

QUERY_EXPANSION_USER_TEMPLATE = """\
Research intent: "{intent}"

Expand this into optimal academic search terms. Respond with the JSON object
only.
"""

# --------------------------------------------------------------------------- #
# Module 3 — single-paper analysis                                            #
# --------------------------------------------------------------------------- #
ACADEMIC_ANALYST_SYSTEM_PROMPT = """\
You are a senior machine-learning researcher and rigorous peer reviewer with
15+ years of experience across ML conferences (NeurIPS, ICML, ICLR, CVPR, ACL).

Non-negotiable rules:
1. GROUNDING — every claim must be supported by the provided paper text.
   If the text does not contain the answer, write exactly
   "Not specified in the paper" for that field or use an empty list.
   NEVER fabricate datasets, numbers, citations or model names.
2. PRECISION — use the paper's own terminology; keep numbers with their units
   and benchmark context (e.g. "+3.2 mIoU on ADE20K vs. Mask2Former").
3. CRITICAL RIGOUR — limitations must be substantive (compute cost,
   generalisation, missing ablations, dataset bias, theoretical gaps), not
   stylistic nitpicks.
4. OUTPUT FORMAT — respond with a single STRICT JSON object. No markdown
   fences, no trailing commentary, no keys outside the requested schema.
"""

_CONTRIBUTION_JSON = """\
{
  "one_line_summary": "single sentence stating the core contribution",
  "motivation": "what problem context motivates this work and why prior approaches fall short",
  "novelty": "what is genuinely new relative to the closest prior art",
  "pain_points_addressed": ["pain point of baselines solved by this work", "..."]
}"""

_METHODOLOGY_JSON = """\
{
  "method_category": "one of: Diffusion Generative | Autoregressive / LLM | Discriminative / Supervised | Self-Supervised | Reinforcement Learning | Graph Neural Network | Optimization / Training Method | Systems & Efficiency | Multimodal | Other",
  "architecture": "model / system architecture, key components and data flow",
  "loss_functions": "training objectives and loss terms (state 'Not specified in the paper' if absent)",
  "training_techniques": "optimiser, schedules, data pipelines, augmentation, tricks",
  "key_assumptions": ["assumption the method relies on", "..."]
}"""

_EXPERIMENTS_JSON = """\
{
  "datasets": ["benchmark names used"],
  "metrics": ["evaluation metrics reported"],
  "baselines": ["methods compared against"],
  "sota_improvement": "quantified improvement over prior SOTA, or 'Not specified in the paper'",
  "main_results_summary": "2-3 sentence narrative of the headline results"
}"""

_CRITIQUE_JSON = """\
{
  "strengths": ["main strengths"],
  "limitations": ["substantive limitations: compute cost, generalisation, weak evaluation, bias, ..."],
  "future_directions": ["concrete follow-up research directions"]
}"""

#: JSON fragment schema shown to the LLM for each chain aspect.
ASPECT_JSON_SCHEMAS = {
    "contribution": _CONTRIBUTION_JSON,
    "methodology": _METHODOLOGY_JSON,
    "experiments": _EXPERIMENTS_JSON,
    "critique": _CRITIQUE_JSON,
}

_TASK_PREFIX = (
    "Analyse the paper below and answer ONLY the requested aspect.\n"
    "Respond with a single JSON object using EXACTLY this schema:\n"
)

CONTRIBUTION_USER_TEMPLATE = _TASK_PREFIX + """\
ASPECT: Core Contribution & Novelty.
Focus on what is new, the motivation, and which baseline pain points it solves.
Schema:
""" + _CONTRIBUTION_JSON + "\n\n{context}"

METHODOLOGY_USER_TEMPLATE = _TASK_PREFIX + """\
ASPECT: Technical Methodology Breakdown.
Focus on architecture, loss functions, training recipe and key assumptions.
Schema:
""" + _METHODOLOGY_JSON + "\n\n{context}"

EXPERIMENTS_USER_TEMPLATE = _TASK_PREFIX + """\
ASPECT: Experimental Validation.
Focus on datasets, metrics, baselines and quantified SOTA improvements.
Schema:
""" + _EXPERIMENTS_JSON + "\n\n{context}"

CRITIQUE_USER_TEMPLATE = _TASK_PREFIX + """\
ASPECT: Critical Thinking / Critique.
Act as a tough but fair reviewer: substantive strengths, limitations and
future directions. Schema:
""" + _CRITIQUE_JSON + "\n\n{context}"

SINGLE_PASS_USER_TEMPLATE = (
    "Analyse the paper below across ALL FOUR aspects.\n"
    "Respond with a single JSON object using EXACTLY this schema:\n"
    "{\n"
    '  "core_contribution": ' + _CONTRIBUTION_JSON.strip() + ",\n"
    '  "methodology": ' + _METHODOLOGY_JSON.strip() + ",\n"
    '  "experiments": ' + _EXPERIMENTS_JSON.strip() + ",\n"
    '  "critique": ' + _CRITIQUE_JSON.strip() + ",\n"
    '  "overall_assessment": "one-paragraph verdict on significance and rigor",\n'
    '  "confidence": "high | medium | low"\n'
    "}\n\n{context}"
)

# --------------------------------------------------------------------------- #
# Module 4 — cross-paper synthesis                                            #
# --------------------------------------------------------------------------- #
SYNTHESIS_SYSTEM_PROMPT = """\
You are a distinguished ML professor writing a survey-grade literature review.
Rules:
1. Ground every statement in the provided per-paper analyses; do not invent
   papers, numbers or chronology.
2. Structure the review exactly with these Markdown sections:
   ## 1. Overview & Scope
   ## 2. Field Evolution Timeline
   ## 3. Method Landscape
   ## 4. Comparative Analysis
   ## 5. Open Challenges
   ## 6. Future Directions
   ## 7. Conclusion
3. In "Field Evolution Timeline", order papers chronologically and explain the
   lineage: which idea enabled which, what bottleneck each work removed.
4. In "Method Landscape", group papers by method category and contrast the
   design philosophies.
5. Respond with STRICT JSON only — no markdown fences around the JSON:
{
  "review_markdown": "the full review in Markdown",
  "evolution_timeline": ["2020 · PaperA — one-line role in the lineage", "..."],
  "open_challenges": ["challenge", "..."],
  "future_directions": ["direction", "..."]
}
"""

SYNTHESIS_USER_TEMPLATE = """\
Review topic: "{topic}"

Per-paper analyses (JSON):
{digest}

Write the literature review now. Remember: strict JSON only.
"""

# --------------------------------------------------------------------------- #
# Renderers                                                                   #
# --------------------------------------------------------------------------- #
#: Priority order in which sections are packed into the analysis context.
_SECTION_PRIORITY = [
    "abstract", "method", "experiments", "results", "introduction",
    "limitations", "conclusion", "discussion", "related_work", "background",
]


def render_analysis_context(content: PaperContent, char_budget: int) -> str:
    """Render a :class:`PaperContent` into a compact LLM context string.

    Sections are packed by analytical priority until ``char_budget`` is
    exhausted; figures, algorithms and formulas are appended last.

    Args:
        content: Parsed paper content.
        char_budget: Maximum approximate character budget.

    Returns:
        Plain-text context block for analysis prompts.
    """
    md = content.metadata
    header_lines = [f"TITLE: {md.title}"]
    if md.authors:
        header_lines.append("AUTHORS: " + ", ".join(md.authors[:15]))
    venue = md.venue or "unknown venue"
    header_lines.append(f"VENUE/YEAR: {venue} / {md.year or 'unknown'}")
    if md.arxiv_id:
        header_lines.append(f"ARXIV: {md.arxiv_id}")
    parts: List[str] = ["\n".join(header_lines)]

    budget = max(1000, char_budget - parts[0].__len__())
    per_section = max(800, budget // 6)
    packed = 0
    for name in _SECTION_PRIORITY:
        text = content.sections.get(name, "").strip()
        if not text:
            continue
        if packed >= budget:
            break
        room = min(per_section, budget - packed)
        if len(text) > room:
            text = text[:room] + " [...]"
        parts.append(f"\n===== SECTION: {name.upper()} =====\n{text}")
        packed += len(text)

    if content.figures:
        caps = "\n".join(
            f"- {f.figure_id}: {f.caption[:220]}" for f in content.figures[:10]
        )
        parts.append("\n===== KEY FIGURE/TABLE CAPTIONS =====\n" + caps)
    if content.algorithms:
        algos = "\n".join(
            f"- {a.algorithm_id}: {a.caption} | {a.pseudocode[:300]}"
            for a in content.algorithms[:4]
        )
        parts.append("\n===== ALGORITHMS =====\n" + algos)
    if content.formulas:
        formulas = "\n".join(
            f"- ${f.latex[:200]}$" for f in content.formulas[:15]
        )
        parts.append("\n===== KEY EQUATIONS (LaTeX) =====\n" + formulas)
    return "\n".join(parts)


def render_synthesis_digest(topic: str, reports: List[AnalysisReport]) -> str:
    """Render analysis reports into a compact digest for the synthesis prompt.

    Args:
        topic: Review topic.
        reports: Completed per-paper analyses.

    Returns:
        A compact string representation consumed by
        :data:`SYNTHESIS_USER_TEMPLATE`.
    """
    lines: List[str] = []
    ordered = sorted(reports, key=lambda r: (r.year or 9999, r.title))
    for i, r in enumerate(ordered, start=1):
        lines.append(
            f"[{i}] title={r.title!r} year={r.year} venue={r.venue or 'unknown'} "
            f"citations={r.citation_count if r.citation_count is not None else 'n/a'}"
        )
        lines.append(f"    category={r.methodology.method_category}")
        lines.append(f"    contribution={r.core_contribution.one_line_summary}")
        lines.append(f"    sota={r.experiments.sota_improvement}")
        lines.append(
            f"    limitations={' ; '.join(r.critique.limitations[:3])}"
        )
        lines.append("")
    digest = "\n".join(lines)
    return f"Total papers: {len(reports)}\n\n{digest}"
