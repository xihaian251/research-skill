# ML Paper Analyst — Agent Skill

生产级、模块化的机器学习文献调研 Agent Skill。一键完成：**跨源检索 → 全文解析 → 深度学术分析 → 多文献综合对比 → 标准化导出**（Literature Review + 对比矩阵 + BibTeX）。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLPaperAnalystSkill.run()                    │
├─────────────────┬─────────────────┬────────────────┬────────────────┤
│  ① Searcher     │  ② Extractor    │  ③ Analyzer    │ ④ Synthesizer  │
│                 │                 │                │                │
│ arXiv API       │ ar5iv HTML      │ Prompt Chain:  │ 对比矩阵(MD)   │
│ Semantic Scholar│ arXiv HTML      │ 1.Contribution │ BibTeX 导出    │
│ HF Daily Papers │ PDF (PyMuPDF)   │ 2.Methodology  │ 文献综述报告   │
│ PWC (legacy)    │ 兜底: metadata  │ 3.Experiments  │ 演进时间线     │
│                 │                 │ 4.Critique     │                │
│ 检索词扩展       │ 章节切分/图表    │ 严格JSON+修复   │ LLM / 模板兜底 │
│ 年份/顶会/引用   │ 公式/伪代码      │ 启发式兜底      │                │
└─────────────────┴─────────────────┴────────────────┴────────────────┘
        共享: HTTPClient(限流+指数退避) · DiskCache(TTL) · LLMBackend
```

## 项目结构

```
ml_paper_analyst/
├── manifest.json      # Agent 工具调用 Schema (Function Calling 定义)
├── skill.yaml         # Skill 元数据 (YAML 注册格式)
├── config.py          # API Key / 限流 / 并发 / 超时 / 缓存配置
├── schemas.py         # Pydantic 数据结构 (PaperMetadata/AnalysisReport/...)
├── prompts.py         # 学术分析 System/User Prompt 模板链
├── tools.py           # 核心执行逻辑 (4 个模块的完整实现)
├── skill.py           # 入口类 MLPaperAnalystSkill (一键 Pipeline)
├── __main__.py        # CLI: python -m ml_paper_analyst "topic"
├── requirements.txt
└── tests/test_smoke.py
```

## 安装

```bash
pip install -r requirements.txt      # aiohttp, pydantic, bs4, lxml, PyMuPDF, openai
```

Python ≥ 3.9。

## 配置（全部可选）

| 环境变量 | 说明 |
|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | 提高 S2 限流配额、启用稳定 enrichment |
| `OPENAI_API_KEY` | 启用 LLM 深度分析；缺省自动降级为启发式抽取分析 |
| `OPENAI_BASE_URL` | 指向任意 OpenAI 兼容网关（vLLM / Ollama / Azure 代理） |
| `MLPA_LLM_MODEL` | 模型名，默认 `gpt-4o-mini` |
| `MLPA_ANALYSIS_MODE` | `single`（默认，1 次调用）或 `chain`（4 步提示词链） |
| `MLPA_CACHE_DIR` / `MLPA_OUTPUT_DIR` | 缓存目录（默认 `~/.cache/ml_paper_analyst`）/ 产物目录 |
| `MLPA_MAX_CONCURRENCY` / `MLPA_REQUEST_TIMEOUT` / `MLPA_MAX_RETRIES` | 并发 / 超时 / 重试次数 |

**无任何 Key 也能运行**：检索、解析、启发式分析与模板综述全部可用。

## 快速开始

### 一键 Pipeline

```python
from ml_paper_analyst import MLPaperAnalystSkill

skill = MLPaperAnalystSkill()  # 自动读取环境变量
result = skill.run_sync(
    topic="diffusion models for text-to-3D",
    max_results=8,
    years=[2022, 2026],
    venue_filter=["CVPR", "ICLR", "NeurIPS"],
    min_citations=50,
)
print(result.review.review_markdown)   # 完整文献综述 (Markdown)
print(result.review.bibtex_entries)    # BibTeX 列表
print(result.stats)                    # 各阶段耗时统计
# 产物已写入 output_dir:
#   <slug>_literature_review.md / _comparison_matrix.md
#   <slug>_references.bib / _full_result.json
```

### 分步调用（Agent 工具粒度）

```python
import asyncio
from ml_paper_analyst import (
    search_ml_papers, fetch_and_parse_paper,
    analyze_single_paper, synthesize_literature_review,
)

async def main():
    papers = await search_ml_papers(
        "score distillation text-to-3d",
        max_results=5, years=[2022, 2026], venue_filter=["CVPR"],
    )
    reports = []
    for p in papers:
        content = await fetch_and_parse_paper(p.arxiv_id or p.paper_id)
        reports.append(await analyze_single_paper(content))
    review = await synthesize_literature_review("Text-to-3D via distillation", reports)
    print(review.review_markdown)

asyncio.run(main())
```

### CLI

```bash
python -m ml_paper_analyst "diffusion text to 3d" \
    -n 6 --years 2022 2026 --venues NeurIPS ICLR CVPR \
    --min-citations 20 --analysis-mode chain -v
```

## 在 Agent 中注册

`manifest.json` 已包含全部 5 个工具的 JSON Schema，可直接映射为 Function Calling：

```python
import json, asyncio
from ml_paper_analyst import MLPaperAnalystSkill

skill = MLPaperAnalystSkill()
manifest = skill.tool_manifest()
tools_for_llm = [
    {"type": "function", "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["parameters"],
    }}
    for t in manifest["tools"]
]

async def dispatch(name: str, args: dict):
    if name == "search_ml_papers":
        from ml_paper_analyst import search_ml_papers
        papers = await search_ml_papers(**args)
        return [p.model_dump() for p in papers]
    if name == "run_ml_literature_analysis":
        result = await skill.run(**args)
        return {
            "papers": [p.title for p in result.papers],
            "review_markdown": result.review.review_markdown
                               if result.review else None,
            "artifacts": result.artifacts,
        }
    raise ValueError(f"unknown tool {name}")

# 将 tools_for_llm + dispatch 接入你的 Agent 框架
# (LangChain / Claude function calling / OpenAI tools 均可)
```

**注入宿主 Agent 自带的大模型**（推荐，省一份 Key）：

```python
from ml_paper_analyst import MLPaperAnalystSkill, LLMBackend

class AgentLLM(LLMBackend):
    async def complete(self, system, user, *, temperature=0.2, max_tokens=3000):
        return await my_agent_model.chat(system + "\n" + user)

skill = MLPaperAnalystSkill(llm_backend=AgentLLM())
```

## 核心设计

- **检索词扩展**：LLM 优先；无 LLM 时使用内置 ML 领域分类法（如 `diffusion text to 3d` → `score distillation sampling / DreamFusion / Gaussian Splatting / differentiable rendering ...`），arXiv 检索以 OR 组合、S2 以原始意图检索保证精确率。
- **过滤**：arXiv 通过 `submittedDate` 区间过滤；顶会过滤内置别名表（`NIPS≈NeurIPS≈Advances in Neural Information Processing Systems`）并解析 arXiv comment 中的 "Accepted to CVPR 2024"；引用量阈值触发 S2 batch API enrichment（`ARXIV:<id>`）。
- **解析**：ar5iv 与 arXiv 原生 HTML 均为 LaTeXML 产物，统一解析（章节层级、`figcaption`、`math alttext` 公式、算法浮动体、机构信息）；失败回退官方 PDF + PyMuPDF + 标题启发式切分；最终兜底 metadata-only（仅摘要），流程永不中断。
- **分析**：严格 JSON 输出 + 平衡括号提取 + 一次"修复重试"；`chain` 模式四步链每个 aspect 独立降级；无 LLM 时启发式抽取（已知数据集/指标/基线名词表 + 句式规则），并以 `confidence=low` 显式标注。
- **健壮性**：per-host 限流（arXiv 3.2s 间隔）、指数退避 + 抖动 + `Retry-After` 遵从、`asyncio.Semaphore` 并发上限、TTL 磁盘缓存（检索 7 天 / 正文 30 天）、缓存损坏自动忽略。
- **注意**：PapersWithCode 已于 2025 年停止服务，`paperswithcode` 源保留但会优雅失败；Hugging Face Daily Papers 作为社区热度维度可用。

## 测试

```bash
pip install pytest pytest-asyncio
pytest tests/ -v          # 离线冒烟测试（不访问网络）
```

集成测试（真实网络，可选）：

```bash
python -c "
import asyncio
from ml_paper_analyst import fetch_and_parse_paper
content = asyncio.run(fetch_and_parse_paper('2301.01234'))
print(content.extraction_source, list(content.sections))
"
```

## 已知约束

1. `synthesize_literature_review` 返回结构化 `LiteratureReview` 对象（Markdown 文本在其 `review_markdown` 字段），便于 Agent 直接消费结构化字段。
2. S2 未鉴权接口共享限流池，偶发 429 时指数退避会自动处理；重调研任务建议配置 `SEMANTIC_SCHOLAR_API_KEY`。
3. 1990 年前的 arXiv 论文、非 arXiv 且无 DOI 的论文无法获取全文。

## License

MIT
