# Plan self-review

Status: proposed plan; author decisions pending. This is not an independent
scientific review and does not certify unwritten material.

## Findings resolved in the plan

1. **Institution attribution:** the folder/uploader suggests IIT Madras. The
   official course page identifies IIT Hyderabad and Prof. Sayantee Jana.
   Use the official course attribution throughout.
2. **Shuffled playlist:** the first entry is Factor Analysis 2. Keep original
   positions and a separate reconstructed teaching sequence. Do not label the
   latter primary-source verified until that check is complete.
3. **Overlapping chapters:** clustering chapters separate linkage definitions,
   Ward geometry and dendrogram interpretation, and centroid optimization.
   Regression chapters separate estimation, influence, design encoding,
   dependent errors, distribution checks, conditioning, and variance modeling.
4. **Prediction versus inference versus causality:** selection uses held-out
   predictive evaluation; diagnostic tests require their actual assumptions;
   regression discontinuity requires identification conditions beyond a fitted
   regression. Keep this distinction explicit in the result statements.
5. **Programming conventions:** the user explicitly requested conversion of all
   R examples to Python. Preserve estimators and interpretation, record changed
   defaults, and explain any example that cannot be faithfully converted.

## Remaining limits

- Only the first teaching transcript has been read. The rest of the outline
  is a mathematical proposal based on upload titles, not a claim about exact
  lecture content or section timestamps.
- The official course page's syllabus link resolves to undefined.pdf, but its
  embedded Svelte server-rendered data contains the full lesson array. All 39
  planned video IDs were subsequently verified against that primary array;
  evidence is retained in sources/official-lecture-order.json.
- The introductory transcript has potential source/ASR errors: it excludes
  irrational values while discussing the real line, and groups count variables
  with qualitative variables. Correct the definitions in the notes; inspect
  source frames/audio before attributing the exact errors to the lecturer.
- The introductory retail anecdotes and football example have not been fact
  checked. Omit them or independently verify them before reuse.
- No dataset availability or optional R reference environment has been verified yet. Do not
  describe a laboratory as reproduced or executable until an actual run passes.
- All external asymptotic and diagnostic-test citations need source verification
  at drafting time. No bibliographic page numbers are guessed in this plan.
- An 18-chapter plan with a provisional 216-page budget and 108 exercises is
  feasible as a production target, not a completed-content inventory.

Gate A: coherent specification proposed. Gate B: exact core statements and
notation proposed. Gate C: distinct chapter purposes and backward dependencies
proposed. Gate D: two core sections and a laboratory per chapter proposed;
source-specific refinements remain necessary after sequential source reading.
