---
name: paper-read
description: Read, analyze, and write structured Chinese paper notes for robotics, VLA, embodied AI, diffusion/transformer, and related ML papers. Use when Codex is asked to read a paper PDF/arXiv page/project page, summarize an existing paper folder, create or update notes under Paper/, compare papers, or produce the user's established paper-reading format with background, contributions, method, experiments, limitations, "灵魂三问", and personal summary.
---

# Paper Read

## Overview

Use this skill to turn a paper, PDF, project page, or existing paper folder into a clear Chinese research note matching the style used in `Paper/`.

Before writing, inspect the target folder for the PDF, existing notes, images, and code-reading notes. If source metadata/results may be stale, verify against primary paper/project/arXiv sources.

## Workflow

1. Gather sources:
   - Prefer the local PDF and existing notes in the target paper folder.
   - If the folder only has a note, use it as the style and factual baseline.
   - If asked to read a new or current paper, verify title, authors, venue/date, arXiv version, project page, and code links from primary sources.

2. Create or select the paper folder:
   - For a new paper, create a dedicated folder under `Paper/` before writing the note.
   - Name new folders as `<YYMMDD>_<ShortName>_<OrgOrLab>_<PaperYear>`, where `<YYMMDD>` is the naming date, for example `260616_HumDex_USC-PSI_2026`.
   - Put the main note, PDF, images, and auxiliary notes in that folder together.
   - If the user already provided a folder or an existing folder clearly matches the paper, reuse it.

3. Extract the paper:
   - Read title/abstract/introduction for problem framing.
   - Read method figures, architecture sections, training details, experiments, ablations, and limitations.
   - Capture exact metrics, datasets, model sizes, robot platforms, and baseline names.
   - Keep equations only when they explain the core mechanism; paraphrase the intuition immediately after.

4. Write the note:
   - Use Chinese by default.
   - Follow the local seven-part structure unless the source folder shows a stronger nearby pattern.
   - Preserve important English terms in parentheses on first use.
   - Make the core idea obvious to a robotics/VLA researcher who wants to decide whether the paper matters.

5. Calibrate claims:
   - Separate author-claimed contributions from your interpretation.
   - Avoid hype; state the evidence that supports each claim.
   - If a number comes from a table/figure, name the table/figure when useful.
   - Highlight limitations, assumptions, and deployment gaps.

6. Final check:
   - Ensure the note has a useful title, metadata block, and links.
   - Ensure all tables render in Markdown.
   - Ensure "灵魂三问" answers are concise, comparative, and evidence-driven.
   - If updating an existing note, keep unrelated user content intact.

## Style Reference

Read `references/note-style.md` when creating or substantially rewriting a note. It contains the local structure, section expectations, naming conventions, and quality bar extracted from the existing `Paper/AssemLM`, `Paper/PI05`, `Paper/PI07`, `Paper/HumDex`, and `Paper/Automic Vla` notes.

## Output Conventions

- Save one main note per paper folder as `<ShortName>_<OrgOrLab>_<PaperYear>.md` by default.
- Always create or reuse a matching paper folder under `Paper/`; do not leave new paper notes directly under `Paper/`.
- Use the date suffix on newly created folder names, not on the main note filename, unless the user explicitly asks for mirrored names.
- Prefer Markdown tables for experiment summaries and ablations.
- Add extra sections after "个人总结" only when the paper has a recurring concept worth preserving, such as metadata taxonomy, code walkthrough notes, or framework comparisons.
