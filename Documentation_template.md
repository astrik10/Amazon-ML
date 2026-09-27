# Business Entity Resolution — Methodology

## Methodology

The pipeline resolves Source-1 business records against Source-2 and
Source-3 in five stages: (1) text normalization, (2) multi-stage blocking
to generate a tractable candidate set per Source-1 entity, (3) ground-truth
label attachment for the training candidates, (4) numeric feature
engineering over every candidate pair, and (5) a classical supervised
classifier that outputs a match probability, thresholded using a held-out,
entity-level validation split. The same normalization, blocking and feature
code paths run identically at train and test time so behavior is
consistent and reproducible.

## Candidate Generation / Blocking

Rather than scoring every possible Source-1 × (Source-2 ∪ Source-3) pair,
a set of inverted-index blocking rules is applied over the Source-2/
Source-3 pool and a candidate is retained if it is returned by **any**
enabled rule (recall-favoring union):

1. **Exact normalized country** — groups records sharing an identical
   normalized country string.
2. **Shared business-name tokens** — tokens of length ≥ `MIN_TOKEN_LENGTH`
   from the normalized name.
3. **Shared address tokens** — tokens of length ≥ `MIN_TOKEN_LENGTH` from
   the normalized address. Enabled via `config.ENABLE_ADDR_TOKEN_BLOCKING`.
4. **Character n-gram overlap on the name** — 4-character shingles, which
   catch near-duplicate names blocking misses on token boundaries (e.g.
   minor misspellings). Enabled via `config.ENABLE_NGRAM_BLOCKING`.
5. **Postal/PIN code overlap** — a postal-like code heuristically extracted
   from the raw address (a run of 4–8 digits), when present in both
   records; weighted higher than the other rules since it's a low-noise
   signal.

The candidate-side inverted index (built from Source-2 + Source-3) is
built once per run and reused across every batch of Source-1 entities
matched against it, rather than being rebuilt per split — the pool doesn't
change between the train split, the validation split, and the final
training run, so there's no reason to redo that work each time.

Any single key (a token, n-gram, or postal code) shared by more than
`config.MAX_POSTING_LIST_SIZE` candidate records is dropped from that
rule before matching — a value that common carries almost no pruning
signal, and keeping it in would mean matching a large fraction of the
whole pool against it for no benefit. Address-token and n-gram blocking
are the most expensive rules at large record counts and are off by
default; they can be re-enabled per dataset size if candidate recall in
the validation report indicates the extra signal is needed.

Every rule that matches a given candidate contributes to a simple integer
score for that candidate. If, after taking the union across all enabled
rules, a Source-1 entity has more than `MAX_CANDIDATES_PER_ENTITY` (60 by
default) candidates, only the highest-scoring ones are kept — a soft cap
for tractability, applied only when blocking is unusually permissive for a
given entity, and it never removes a candidate that no rule would have
retained in the first place.

## Feature Engineering

All features are numeric and missing-value-safe (missing text normalizes
to the empty string, which every similarity function handles explicitly).

**Name features**: exact match, character-set (Jaccard-style) similarity,
Levenshtein normalized similarity, Jaro-Winkler similarity, token Jaccard
similarity, raw token overlap count, TF-IDF cosine similarity, and absolute
length difference.

**Address features**: the same family (exact match, character similarity,
Levenshtein similarity, token Jaccard, token overlap, TF-IDF cosine, length
difference), plus an explicit postal-code match flag.

**Other**: country exact match, and a flag for whether the candidate comes
from Source-3 vs. Source-2 (since the two sources may have systematically
different data quality).

TF-IDF vectorizers are fit fresh on the relevant record pool for each
pipeline run (the validation run's train+val entities, and separately the
final test run), so no vocabulary leaks between training and test. The
TF-IDF cosine similarity for a whole batch of candidate pairs is computed
as a single vectorized sparse-matrix operation rather than pair-by-pair,
since TF-IDF vectors are already L2-normalized and cosine similarity
reduces to a dot product.

## Model Architecture

The default model is scikit-learn's `LogisticRegression` with
`class_weight="balanced"` to counteract the heavy non-match/match class
imbalance inherent to entity resolution (most candidate pairs are
non-matches even after blocking). It is fast, well-calibrated for
`predict_proba()`-based thresholding, and easy to reason about. A
`RandomForestClassifier` baseline is available as a drop-in alternative via
`config.MODEL_TYPE`. Both are BSD-licensed, classical (non-deep-learning)
models with no parameter-count concerns relative to the 8B ceiling.

## Validation

The Source-1 **training entities** (not raw rows or pairs) are split into a
train portion and a validation portion using a seeded, deterministic
shuffle. Because the split happens at the entity level before any
candidate generation, no record belonging to one Source-1 entity can appear
in both the train and validation portions. The model used to score the
validation split is trained only on the train-portion's candidate pairs;
the final model used for test inference is retrained afterward on 100% of
the training data.

Metrics reported: pair-level precision/recall/F0.5 (F0.5 = 1.25·P·R /
(0.25·P + R), weighting precision higher than recall), and — matching how
the challenge is scored — a **macro-averaged entity-level F0.5** where each
Source-1 entity's predicted match set is compared against its true match
set, with explicit singleton handling (an entity with no true matches
scores 1.0 if predicted empty, 0.0 if predicted non-empty).

## Threshold Selection

Every threshold in `{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}` is applied
to the validation-split predicted probabilities. The threshold that
maximizes **macro entity-level F0.5 on the validation split** is selected
and reused, unchanged, for test inference. No threshold tuning happens on
test data.

## Output Generation

`output/candidate_pairs.tsv` records, for every test Source-1 entity, the
exact set of Source-2/Source-3 IDs that were sent to the model (i.e. the
post-blocking, pre-threshold candidate set). `output/matching_results.tsv`
records, for every test Source-1 entity, the subset of those candidates
whose predicted probability met or exceeded the chosen threshold. Before
either file is written, the pipeline asserts every ID in
`matching_results.tsv` also appears in that entity's `candidate_pairs.tsv`
row.

## Reproducibility

A single fixed random seed (`config.RANDOM_SEED = 42`) drives both the
entity-level train/validation split and the model's internal randomness
(e.g. Random Forest bootstrapping). All dependencies are pinned in
`requirements.txt`. Running `python -m src.main` against the same
`dataset/` contents will always produce the same outputs. No dataset row
counts or country lists are hard-coded anywhere in the code.

## Fair Play

This project uses **only** the data provided under `dataset/train/` and
`dataset/test/`. No internet access, external APIs, geocoding services, or
government/business-lookup databases were used at any stage of
development, feature engineering, model training, or inference.