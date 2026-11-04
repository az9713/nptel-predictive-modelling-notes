# Lecture-note production workflow

This project adapts the user's NPTEL quantum-computation and high-performance
scientific-computing workflows, and the Stanford CS229 study-edition structure.
Those repositories were inspected during setup; their validation results do not
validate this course.

1. Inventory the exact playlist and verify course identity. Preserve upload
   order separately from a reconstructed teaching order. Record missing items,
   duplicates, retrieval failures, and unresolved ordering questions.
2. Prepare specification, notation, result dependencies, and a detailed chapter
   plan using stem-textbook. Resolve routine decisions autonomously under the
   user's explicit instruction to complete the course without intermediate gates.
3. Acquire timestamped English captions; if unusable or unavailable, use local
   audio transcription with the limitation recorded. Read each assigned source
   sequentially. Retain raw text, media, and source frames privately.
4. Draft one section at a time with rigorous-explainer. Begin from a motivating
   problem, define notation and assumptions, derive the claims, interpret the
   result, and verify worked examples. Inspect frames where text is ambiguous.
5. Convert every R demonstration to Python, per the user's explicit instruction.
   Inventory each R example with a source timestamp and conversion status.
   Execute every laboratory and printed numerical result. Match source inputs
   and statistical conventions; explain default-dependent differences. Record
   missing data, unavailable methods, and unresolved conversions individually.
   Do not claim a fresh R/Python comparison without executing both. Source R
   output, an independent calculation, and a fresh R run are different evidence.
6. Invoke science-editor in separate reviewer contexts for two independent
   chapter review rounds. Repair findings and obtain a current-content verdict;
   bind acceptance to the reviewed body digest. Self-checks do not count as
   independent review. Source fidelity and mathematical correctness are separate
   review dimensions.
7. Assemble the offline HTML reader, source map, study guide, and code laboratory.
   Check local links, timestamps, mathematics, controls, desktop/mobile layouts,
   figures, and printing. Apply relevant rigorous-explainer validators.
8. Verify a source-free export, with reproducible build and code checks. Keep
   source coverage, editorial acceptance, program execution, browser checks,
   and publication verification as separate evidence records.
9. Prepare a public-ready package excluding private sources and internal logs.
   Publish only if the user requests it; verify the actual served files then.

All 39 teaching recordings are covered in 18 authored chapters, with 108 worked
solutions and 18 executable laboratories. Current-content review acceptance,
execution results, reader checks and export checks are recorded separately in
completion.json and validation/. The GitHub reading edition is published at
https://az9713.github.io/nptel-predictive-modelling-notes/ ; served-file and live
reader checks are recorded in validation/publication-checks.json.
