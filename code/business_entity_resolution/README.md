# Business Entity Resolution

Match every Source-1 business record with zero, one, or multiple records
from Source-2 and Source-3, using **only** the provided datasets — no
internet access, external APIs, geocoding services, or government/business
lookup databases.

---

## 1. What this project does

Given three "sources" of business records (name, address, country) and a
training file of known matches, the pipeline:

1. Normalizes text (names, addresses, countries).
2. Generates a manageable set of **candidate** Source-2/Source-3 records for
   every Source-1 record (blocking), instead of comparing every pair.
3. Builds numeric similarity features for each candidate pair.
4. Trains a classical ML classifier (Logistic Regression by default) to
   predict match probability.
5. Selects a probability threshold using a held-out validation split.
6. Applies the trained model + threshold to the test set and writes the two
   required output files.

Country is treated as an **open-set string** — there is no hard-coded list
of valid countries anywhere in the code.

The full pipeline (~2.2M Source-1 training entities, ~5M+ Source-2, ~5.3M
Source-3 training records; a similarly sized test set) is processed in
fixed-size chunks throughout rather than all at once — see §6 and §9 for
why, and §4 for actual end-to-end timing.

---

## 2. Dataset structure

Place your dataset under `dataset/` (this folder is gitignored — bring your
own data, it is never bundled with the code):

```
dataset/
├── train/
│   ├── train_source1.tsv        # entity_id, business_name, business_address, country
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv   # source1_entity_id, matched_entity_ids (comma-separated)
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are **tab-separated** (`sep="\t"`). `entity_id` values are
prefixed `S1-`, `S2-`, `S3-`; the source of a file is auto-detected from
these prefixes rather than assumed from the file name.

---

## 3. Installing dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Running training + validation + test inference

Everything runs in one command from the project root:

```bash
python -m src.main
```

This will:

- Load and normalize `dataset/train/*`.
- Run an **entity-level** train/validation split (see §5), sweep
  probability thresholds, and print a validation report to the console.
- Train the final model on **all** training data and save it to
  `models/entity_resolution_model.joblib`.
- Load and normalize `dataset/test/*`, generate candidates, score them, and
  apply the threshold chosen during validation.
- Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

**Timing (reference run, ~12.5M total train records, ~11M total test
records, on a standard laptop):** end-to-end run took roughly 3 hours
15 minutes, logging progress per chunk throughout (`STEP 1/5` through
`STEP 5/5`). Actual time depends heavily on available RAM and CPU.

`src/main.py` has a `SAMPLE_SOURCE1_ROWS` switch at the top for sanity-
checking the pipeline on a small subset before a full run; it must be
`None` for any real/final run, since every Source-1 test entity has to
appear in the output.

---

## 5. How validation works

`src/evaluation.py` splits the **Source-1 training entities** (not
individual pairs/rows) into a train portion and a validation portion, so no
record belonging to the same Source-1 entity leaks across the split. For
each candidate probability threshold in `config.THRESHOLD_CANDIDATES`
(0.30 … 0.90), it computes:

- pair-level precision / recall / F0.5,
- **entity-level** F0.5, macro-averaged across all validation Source-1
  entities (matching how the challenge scores submissions), with explicit
  singleton handling:
  - true matches empty AND predicted empty → score **1.0**
  - true matches empty AND predicted non-empty → score **0.0**
  - otherwise, standard precision/recall/F0.5 over the two ID sets.

The threshold with the highest macro entity-level F0.5 **on validation
only** is the one used for test inference.

**Reference validation results** (full training set, 1,765,457 train /
441,364 validation entities): candidate recall ≈ 0.280 on both splits;
chosen threshold **0.90**, giving pair-level precision 0.8286 / recall
0.9577 / F0.5 0.8515, and **macro entity-level F0.5 ≈ 0.3513** — this last
number is what the challenge actually scores. The train/validation gap
between pair-level and entity-level F0.5 mostly reflects blocking recall
(see below): an entity with even one missed true match scores lower at
the entity level even when every pair it did see was classified
correctly.

---

## 6. How blocking works

`src/blocking.py` builds inverted indexes over the Source-2/Source-3 pool
**once per run** (`build_candidate_index`) and reuses that index for every
chunk of Source-1 entities matched against it (train split, validation
split, final training run, test run), instead of rebuilding it from
scratch each time — the pool itself doesn't change between those runs.

| Rule | Signal | Default |
|---|---|---|
| Country block | exact normalized country match | on |
| Name token block | shared normalized business-name tokens (≥ `MIN_TOKEN_LENGTH`) | on |
| Address token block | shared normalized address tokens (≥ `MIN_TOKEN_LENGTH`) | off — `config.ENABLE_ADDR_TOKEN_BLOCKING` |
| Name character n-gram block | shared 4-character n-grams of the name (catches typos/abbrev.) | off — `config.ENABLE_NGRAM_BLOCKING` |
| Postal/PIN block | shared postal code heuristically extracted from the address | on, weighted higher |

A candidate is kept if it is retrieved by **any enabled** rule (union, for
recall). Address-token and n-gram blocking add recall but are the most
expensive rules on large record counts, so they're off by default; they
can be flipped on in `config.py` if candidate recall (reported in the
validation output — see §5) needs to be higher, now that the pipeline
processes everything in bounded chunks and has headroom to do so.

Keys shared by more than `config.MAX_POSTING_LIST_SIZE` candidate records
are dropped from a rule before matching. Each rule that fires for a
candidate adds to a simple score; if an entity ends up with more than
`MAX_CANDIDATES_PER_ENTITY` (default 60) candidates, only the highest-
scoring ones are kept.

**Why `generate_candidates` itself is chunked:** matching the *entire*
Source-1 set (millions of entities) against the candidate index in one
merge produces an intermediate hit table that can run into the hundreds
of millions of rows before it's capped down per entity — this exhausted
memory in practice. `blocking.generate_candidates` therefore internally
processes Source-1 in batches of `config.SOURCE1_CHUNK_SIZE` (default
50,000) and concatenates the (much smaller) per-chunk results; callers
don't need to chunk it themselves.

---

## 7. How features are generated

`src/features.py` computes, for every candidate pair, all-numeric features
covering:

- **Name**: exact match, character-set similarity, Levenshtein similarity,
  Jaro-Winkler similarity (via `rapidfuzz`), token Jaccard, token overlap
  count, TF-IDF cosine similarity, length difference.
- **Address**: the same family of features (Levenshtein, Jaccard, overlap,
  TF-IDF cosine, length difference), plus postal-code match.
- **Other**: country exact match, whether the candidate is from Source-3.

TF-IDF vectorizers are fit once per pipeline run (once for the validation
split, once for test inference) over the relevant Source-1+2+3 corpus, so
vocabulary never leaks between the two runs. The TF-IDF cosine similarity
itself is computed for a whole batch of pairs at once via a vectorized
sparse dot product (TF-IDF vectors are L2-normalized, so cosine similarity
reduces to a dot product), rather than one pair at a time.

---

## 8. Which ML model is used

`src/model.py` wraps scikit-learn's `LogisticRegression`
(`class_weight="balanced"`, to handle the natural non-match/match class
imbalance) as the default baseline. `RandomForestClassifier` is available
by setting `MODEL_TYPE = "random_forest"` in `src/config.py`. Both are
BSD-3-Clause licensed and far below the 8B-parameter ceiling — a
permissive license in the same family as MIT/Apache 2.0 (no copyleft,
free commercial/derivative use).

---

## 9. Chunked processing and negative sampling

At the full dataset's scale, three stages that operate over every
Source-1 entity — blocking, feature construction, and model
training/scoring — are all processed in fixed-size batches of
`config.SOURCE1_CHUNK_SIZE` (default 50,000) rather than materializing
the whole dataset's candidate pairs / features at once:

- **Training set construction** (`evaluation._build_training_set`,
  `inference.train_final_model`): for each chunk, every positive
  (true-match) pair is kept, and non-matching pairs are downsampled to at
  most `config.NEGATIVE_SAMPLE_RATIO` (default 10) negatives per positive,
  floored at `config.MIN_NEGATIVES_PER_CHUNK`. This is a standard technique
  for heavily imbalanced classification — the overwhelming majority of
  candidate pairs are non-matches even after blocking, and a
  `class_weight="balanced"` logistic regression doesn't need every one of
  them to characterize the non-match distribution. The accumulated
  training set stays a bounded size regardless of how many Source-1
  training entities there are.
- **Validation scoring and test inference** (`evaluation._score_validation_set`,
  `inference.run_test_inference`): every candidate pair is scored, with no
  sampling of any kind — this only affects what the model is *trained* on,
  never what it's *evaluated* or *applied* to. Each chunk's pair/feature
  table is discarded once that chunk's (much smaller) per-entity results
  have been produced, so memory use stays flat regardless of dataset size.

---

## 10. How the threshold is selected

See §5 — the threshold in `config.THRESHOLD_CANDIDATES` that maximizes
macro entity-level F0.5 on the validation split is stored and reused
unchanged for test inference. It is **never** re-tuned on test data.

---

## 11. How test inference works

`src/inference.py`:

1. Trains a **final** model on 100% of the training pairs (all positives,
   downsampled negatives per §9 — more data than the validation-time
   model, which only sees the train-split portion).
2. Loads and normalizes `dataset/test/*`.
3. Runs the same blocking + feature pipeline on the test data, in chunks
   (§9) — every test candidate is scored, nothing sampled.
4. Keeps candidates at/above the chosen threshold as final matches.
5. Asserts every matched ID is present in that entity's own candidate list
   before writing anything to disk.

---

## 12. Output files

### `output/matching_results.tsv`

```
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

- Exactly one row per test Source-1 entity.
- Empty `matched_entity_ids` for singletons (no matches).
- Comma-separated IDs, no spaces, no duplicates, only S2/S3 IDs.

### `output/candidate_pairs.tsv`

```
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

- The **final** candidate set actually sent to the ML model for that
  entity (i.e. after blocking, before thresholding).
- Every ID in `matching_results.tsv` is guaranteed to appear here too.

---

## 13. Validating a submission

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

(On some systems this is invoked as `python3` instead of `python`,
depending on how Python is installed / aliased.)

This checks (10 rules): every test Source-1 ID appears exactly once in both
files, no duplicate rows, no duplicate IDs within a row, all matched/candidate
IDs exist in test Source-2/Source-3, no S1 IDs appear as matches or
candidates, every matched ID is present in that entity's candidate list, and
both files are correctly tab-separated with the exact required columns.

---

## 14. Fair play

This project uses **only** the files under `dataset/`. No internet access,
external APIs, geocoding services, or government/business-lookup databases
are used anywhere in the pipeline.

---

## 15. Project structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py          # paths, seeds, thresholds, schema, blocking/chunking limits
│   ├── data_loader.py      # TSV loading + schema/source validation
│   ├── preprocessing.py    # text normalization
│   ├── blocking.py         # candidate generation (reusable index, internally chunked)
│   ├── labels.py           # ground-truth parsing / pair labeling
│   ├── features.py         # similarity feature engineering
│   ├── model.py             # LogisticRegression / RandomForest wrapper
│   ├── evaluation.py       # entity-level split, chunked training set, threshold selection
│   ├── inference.py        # final model training + chunked test-time scoring
│   └── main.py              # end-to-end orchestration
├── utils/
│   └── validate_submission.py
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── README.md
└── requirements.txt
```