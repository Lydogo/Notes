---
name: paper-read
description: Read, analyze, and write structured Chinese paper notes for robotics, VLA, embodied AI, diffusion/transformer, and related ML papers. Use when Codex is asked to read a paper PDF/arXiv page/project page, summarize an existing paper folder, create or update notes under Paper/, organize/index the Paper collection, compare papers, or produce the user's established paper-reading format with background, contributions, method, experiments, limitations, "灵魂三问", and personal summary. Emphasize data usage and training flow when they matter to a VLA/robotics paper, and keep the collection catalog in Paper/INDEX.md current.
---

# Paper Read

## Overview

Use this skill to turn a paper, PDF, project page, or existing paper folder into a clear Chinese research note matching the style used in `Paper/`.

Before writing, inspect the target folder for the PDF, existing notes, images, and code-reading notes. If source metadata/results may be stale, verify against primary paper/project/arXiv sources.

The user is currently researching robotics VLA models, with a particular focus on pre-training and data engineering. When a paper touches data scaling, action representation, embodiment alignment, world modeling, visual grounding, trajectory/action supervision, or staged training recipes, make those mechanisms especially clear and connect them naturally to nearby VLA pre-training work. When data is relevant, trace the important path from source to sample, preprocessing, supervision, and training/evaluation use. Prioritize details that explain the method or affect reproducibility; include tensor/action dimensions when they are reported and useful.

## Workflow

1. Gather sources:
   - Prefer the local PDF and existing notes in the target paper folder.
   - If the folder only has a note, use it as the style and factual baseline.
   - If asked to read a new or current paper, verify title, authors, venue/date, arXiv version, project page, and code links from primary sources.

2. Create or select the paper folder:
   - For a new paper, create a dedicated folder under `Paper/` before writing the note.
   - Name new folders as `<YYMMDD>_<ShortName>_<OrgOrLab>_<PaperYear>`, where `<YYMMDD>` is the local date the note is first added to the library. Do not derive it from the PDF timestamp, which can change when a file is copied or downloaded again. Reuse existing folder names as they are.
   - Put the main note, images, and auxiliary notes in that folder together. Treat the PDF as a temporary source file and remove it after the note passes the final check.
   - If the user already provided a folder or an existing folder clearly matches the paper, reuse it.

3. Extract the paper:
   - Read title/abstract/introduction for problem framing.
   - Read method figures, architecture sections, training details, experiments, ablations, and limitations.
   - Capture exact metrics, datasets, model sizes, robot platforms, and baseline names.
   - When data or training is central to the paper's contribution or reproducibility, build a compact data map: relevant sources, sample unit, modalities, transformations, supervision, and role in each training/evaluation stage.
   - For VLA/robotics papers, record reported dimensions when they help explain the method or reproduce it, such as camera setup, state/action dimensions, action horizon, control frequency, coordinate frame, or action encoding. Treat this list as prompts, not a checklist for every paper.
   - Keep equations only when they explain the core mechanism; paraphrase the intuition immediately after.

4. Write the note:
   - Use Chinese by default.
   - Follow the local seven-part structure unless the source folder shows a stronger nearby pattern.
   - Preserve important English terms in parentheses on first use.
   - Include a concise data-focused subsection when data or training materially affects the method, claims, or reproducibility. Use the guidance in `references/note-style.md` in proportion to the paper; for a key missing detail, say "论文未说明" instead of guessing.
   - Make the core idea obvious to a robotics/VLA researcher who wants to decide whether the paper matters.

5. Calibrate claims:
   - Separate author-claimed contributions from your interpretation.
   - Avoid hype; state the evidence that supports each claim.
   - If a number comes from a table/figure, name the table/figure when useful.
   - Highlight limitations, assumptions, and deployment gaps.

6. Final check:
   - Ensure the note has a useful title, metadata block, and links.
   - Ensure all tables render in Markdown.
   - When the note has a data section, check that it covers the paper's important reported data path: source, sample/supervision, material preprocessing, and training/evaluation role. Add dimensions or mixture details when they are reported and relevant; do not expand the section just to fill every possible field.
   - Ensure "灵魂三问" answers are concise, comparative, and evidence-driven.
   - If updating an existing note, keep unrelated user content intact.
   - After the Markdown note is created and checked, remove the source PDF. If the PDF was moved into the paper folder during organization, delete that copy too; keep only the Markdown note and any necessary images or auxiliary notes.
   - When creating a main note or organizing the collection, add or update its entry in `../INDEX.md`. Give each main note one primary direction and a few cross-cutting topic tags; keep code walkthroughs and translations in the index's supplemental-material section.

## Style Reference

Read `references/note-style.md` when creating or substantially rewriting a note. It contains the local structure, section expectations, naming conventions, and quality bar extracted from the existing `Paper/AssemLM`, `Paper/PI05`, `Paper/PI07`, `Paper/HumDex`, and `Paper/Automic Vla` notes.

## Output Conventions

- Save one main note per paper folder as `<ShortName>_<OrgOrLab>_<PaperYear>.md` by default.
- Always create or reuse a matching paper folder under `Paper/`; do not leave new paper notes directly under `Paper/`.
- Do not retain the original PDF after the Markdown note is generated and verified. The PDF is a temporary source artifact, not part of the final paper folder.
- Use the date prefix on newly created folder names, not on the main note filename, unless the user explicitly asks for mirrored names.
- Keep `../INDEX.md` as the collection catalog: one entry per canonical paper note, with one primary direction and a small set of useful topic tags. Link supplemental code notes or translations separately.
- Prefer Markdown tables for experiment summaries and ablations.
- Add extra sections after "个人总结" only when the paper has a recurring concept worth preserving, such as metadata taxonomy, code walkthrough notes, or framework comparisons.
