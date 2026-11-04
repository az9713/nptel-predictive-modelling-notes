# Course production contract

The user explicitly authorized autonomous completion of the entire course,
without intermediate approval requests, using the established NPTEL/Stanford
lecture-note workflow. All source R examples must become Python, and every
remaining reproduction or conversion obstacle must be explained individually.
The user subsequently requested committing and pushing to GitHub in the same
style and structure as the other lecture-note repositories. Public repository
creation and root-level GitHub Pages publication follow that instruction.

Read BOOK-STATE.yaml, EDITOR_DOMAIN.md, WORKFLOW.md, plan/notation.md, and the
applicable chapter metadata before editing. Current counts and reviewed hashes
live in completion.json; old summaries are not evidence of current acceptance.

- Work only inside this project. Preserve other agents' edits and assigned files.
- Use stem-textbook for structure, rigorous-explainer for chapter development,
  and science-editor in independent contexts for two review rounds per chapter.
- Chapters are authored directly as chapters/chNN.html fragments, section by
  section. The builder supplies the shared root-level reader; it generates no
  mathematical exposition. Do not put chapter prose inside generator scripts.
- Read the complete assigned timestamped transcripts. Inspect selected frames
  when equations, code, data, or package defaults are ambiguous. Record limits.
- Prove or properly cite claims, define notation locally, execute every printed
  numerical construction, and verify all six solved exercises per chapter.
- Preserve R estimator and preprocessing conventions in Python. Exact original
  data reproduction, same-input R/Python agreement, and algorithmic equivalence
  are separate claims. Record arbitrary signs, label permutations, RNG streams,
  missing data, nonconvergence, package versions, and partial coverage accurately.
- Keep raw media, captions, R reference environments/scripts, downloaded source
  packages, and working source frames in sources/private, excluded from exports.
- Bind editorial verdicts to the actual chapter fragment SHA-256. A changed
  fragment needs review of its repairs before acceptance is current again.
- Mechanical, numerical, source, scientific, browser, and source-free checks are
  separate evidence. Never fabricate acceptance or human certification.
- Customer-facing surfaces are HTML, dark by default. Internal plans may be
  Markdown. The portable export must rebuild without private inputs or network.
- Execute tools/build_book.py, tools/check_book.py, tools/check_reader.py, and
  tools/check_export.py with the project Python environment at final validation.
