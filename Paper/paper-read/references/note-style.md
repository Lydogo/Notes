# Local Paper Note Style

## Main Structure

Use this structure by default:

```markdown
# 中文短标题：一句话点出论文对象或核心能力

> 原标题：...
> 作者：...
> 机构：...
> 发表：...
> 链接：...
> 开源：...  <!-- if available -->

---

## 一、研究背景与动机
## 二、核心贡献
## 三、方法原理
### 3.1 整体框架
### 3.2 关键技术细节
### 3.3 训练与优化
### 3.4 数据使用与维度追踪
## 四、实验与结果
### 4.1 实验设置
### 4.2 主要结果
### 4.3 消融实验
## 五、局限性与展望
## 六、灵魂三问
## 七、个人总结
```

If the paper is especially dense, add "一句话说清楚" before section one. If the paper has an important conceptual appendix or implementation lesson, add section eight after "个人总结".

## Section Expectations

### Title And Metadata

- Chinese title should be readable, not a literal machine translation.
- Metadata block should include original title, authors, institution when useful, publication/arXiv version/date, paper link, project page, and code link.
- Use `---` after metadata.

### 一、研究背景与动机

- Start with the core problem.
- Name concrete failure modes of prior work.
- End with the paper's goal.
- For robotics/VLA papers, distinguish task planning, action generation, robot embodiment, data scaling, sim-to-real, and long-horizon issues.

### 二、核心贡献

- Use 3-4 numbered contributions.
- Keep each contribution falsifiable: model, dataset, training recipe, benchmark, real-robot result, open-source artifact.
- Avoid restating the abstract verbatim.

### 三、方法原理

- Start with the overall pipeline: inputs, outputs, model backbone, modality flow, and inference loop.
- Put the most distinctive mechanism first in "关键技术细节".
- Explain why the mechanism is different from obvious alternatives.
- Translate equations into intuition immediately after the equation.
- Include implementation/training details only if they help reproduce or judge the paper.
- For VLA/robotics papers, always include "3.4 数据使用与维度追踪" unless the paper is purely conceptual and has no data/training component.

Common robotics/VLA details worth capturing:

- Backbone and parameter scale.
- Visual encoder, language model, action head/expert, tokenizer/action representation.
- Action space, action chunk length, control frequency, robot platform.
- Data sources, filtering/labeling strategy, loss functions, training stages.
- Whether the system is open-loop, closed-loop, hierarchical, end-to-end, or modular.

### 3.4 数据使用与维度追踪

Write this section for the user's VLA algorithm-engineering perspective. Be concrete and compact; prefer tables and explicit shapes over narrative filler. If a detail is absent from the paper, write "论文未说明" or "未报告", not an inferred value.

Start with a data inventory table when there is more than one source:

```markdown
| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| ... | ... | episode / trajectory / transition / chunk | RGB, proprio, language, action | ... | ... | pretrain / finetune / eval | filtering, resampling, normalization... |
```

Then explain these points when available:

- **样本单位**: one training item is an episode, trajectory window, transition, image-language pair, video clip, or action chunk; specify context length and prediction target.
- **视觉数据**: camera count, view names, RGB/depth usage, resolution before/after crop/resize, frame rate, temporal stacking, augmentation, and encoder input shape.
- **语言数据**: instruction/source, template or human annotation, tokenizer, maximum token length, masking, and whether language is used as goal, observation, or supervision.
- **状态与动作**: proprioception fields, state dimension, action dimension, absolute vs delta action, world/base/EEF frame, rotation representation, gripper encoding, action horizon/chunk length, control frequency, and post-processing before robot execution.
- **预处理链路**: filtering, deduplication, synchronization, interpolation, resampling, normalization statistics, discretization/tokenization, padding, masks, coordinate transforms, and embodiment-specific adapters.
- **训练配比**: dataset mixture ratios, sampling temperature, stage-wise usage, loss weights, supervised tokens/actions, frozen vs trainable modules, and whether high-quality robot data is upsampled.
- **评估数据**: train/test split, task instances, random seeds, number of trials, simulator vs real robot split, seen/unseen objects or instructions, and whether demos overlap with training data.

For papers with important dimensions, add a short "维度快照" block:

```markdown
**维度快照**
- Observation: ...
- Language: ...
- State: ...
- Action: ...
- Prediction target: ...
```

The goal is not to over-document every dataset citation; it is to make clear how raw data becomes model input/supervision and which hidden engineering choices affect reproducibility.

### 四、实验与结果

- State datasets, tasks, metrics, baselines, and robot platforms before results.
- In "4.1 实验设置", separate training data from evaluation data; name splits, task counts, trial counts, and any seen/unseen protocol.
- Use tables for headline comparisons and ablations.
- Record exact numbers when available.
- After each table, write the takeaway in 1-3 bullets.
- Identify the single strongest empirical evidence, not just the largest metric.

### 五、局限性与展望

- Split author-stated limitations and your inferred limitations when both are present.
- Prioritize deployment blockers: data cost, latency, error accumulation, perception dependency, embodiment mismatch, sim-to-real, hardware constraints, closed-loop absence.
- Suggest plausible next steps without overclaiming.

### 六、灵魂三问

Use exactly three questions:

1. **它解决了什么问题？**
   Answer by contrasting with prior methods and naming the concrete bottleneck.
2. **为什么这么做？**
   Answer with the core design choice and why alternatives are weaker; for VLA papers, include how the data representation or training data flow supports this choice when it is central.
3. **什么证据最有说服力？**
   Answer with the cleanest experiment, ablation, figure, data scaling result, or real-robot evidence.

Answers should be short paragraphs, not generic bullets.

### 七、个人总结

Use 2-3 numbered points:

1. Core idea in one compact statement.
2. Biggest advantage and biggest weakness.
3. Implication for future research or for the user's likely robotics/VLA interests.

## Tone And Language

- Default language: Chinese.
- Keep key English terms on first mention, e.g. 视觉-语言-动作模型（Vision-Language-Action, VLA）.
- Use confident but bounded language: "说明", "表明", "作者认为", "更像是".
- Prefer concrete mechanism-level explanation over broad praise.
- Keep a researcher's reading voice: clear, slightly opinionated, evidence-aware.

## Naming

- Folder: create one folder per paper under `Paper/`, named `<YYMMDD>_<ShortName>_<OrgOrLab>_<PaperYear>` by default.
- Use `<YYMMDD>` as the PDF file's modification/creation date in local time (check with `stat`). For example, a PDF last modified on June 16, 2026 yields `260616_`.
- Main note filename: use the canonical paper name without the naming-date suffix, `<ShortName>_<OrgOrLab>_<PaperYear>.md`, unless the user explicitly asks for mirrored folder/file names.
- Treat source PDFs as temporary artifacts and remove them after the Markdown note is generated and checked, unless the user explicitly asks to retain the PDF.
- Auxiliary code notes may use names like `openpi05_code_walkthrough.md` or `<paper>_code_learning.md`.

Examples from this repository:

- Existing folders such as `Paper/AssemLM/AssemLM_Fudan_TeleAI_2026.md` are acceptable and should be reused when they clearly match.
- Existing full-pattern folders such as `Paper/HumDex_USC-PSI_2026/HumDex_USC-PSI_2026.md` should not be renamed just to add a date suffix.
- New folders should use the date prefix, e.g. `Paper/260616_GR00T-N1_Nvidia_2025/GR00T-N1_Nvidia_2025.md`.

If choosing for a new paper, prefer the full pattern with date prefix unless the user specifies a shorter folder name.

## Quality Bar

A good note should let the reader answer:

- What new capability or claim does this paper make?
- What is the core technical move?
- What makes the evidence believable?
- Where will it probably fail?
- How does it connect to nearby VLA/robotics papers in this repository?
