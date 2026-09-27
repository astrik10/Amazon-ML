# Documentation_template.md — Business Entity Resolution

Team: TENSOR-TITANS
Challenge: Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Methodology Used

In simple words, our job was to look at business records coming from three
different sources and figure out which ones are actually talking about the
same real business — even when the name is spelled differently, the address
is written in a different way, or some details are missing.

We solved this in **five simple steps**, and the whole thing runs with one
command: `python -m src.main`

1. **Cleaning up the text (Normalization)** — *(`src/data_loader.py`)*
   Business names and addresses are written very differently by different
   people — some write "Rd", some write "Road"; some write "Pvt Ltd", some
   write "Private Limited". So the first thing we do is clean everything up:
   convert to lowercase, remove punctuation and accents, and expand common
   short forms to a standard form. This way, two records that actually mean
   the same thing start looking similar to the computer too.

2. **Shortlisting possible matches (Candidate Generation / Blocking)** — *(`src/blocking.py`)*
   We can't compare every Source-1 record with every single Source-2 and
   Source-3 record — there are millions of records, so that would take
   forever. Instead, for every Source-1 entity, we quickly shortlist a small
   set of realistic possible matches. Full details on how we do this are in
   Section 2.

3. **Attaching the answer key (only for training data)** — *(`src/evaluation.py`)*
   For the training data, we already know the correct answers
   (`train_ground_truth.tsv`). So we label every shortlisted pair as
   1 (actual match) or 0 (not a match), so our model can learn from it.

4. **Turning pairs into numbers (Feature Engineering)** — *(`src/features.py`)*
   Computers can't understand "this name looks similar" the way humans do —
   so we convert every pair of records into a set of numeric similarity
   scores (how similar the names are, how similar the addresses are, etc.).
   Details are in Section 3.

5. **Predicting matches (Model + Threshold)** — *(`src/model.py` and `src/inference.py`)*
   A machine learning model looks at these numbers and gives a probability
   score for each pair — basically "how likely is it that these two records
   are the same business?". We then pick one cut-off point (called a
   threshold) using our validation data, and use that same cut-off point to
   decide the final matches on the test data.

The good part is: the exact same cleaning, shortlisting, and
feature-building code runs during training and during testing — so there is
no unfair advantage or mismatch between the two.

**Why we process data in batches (chunks):** *(controlled by `config.SOURCE1_CHUNK_SIZE` in `src/config.py`, used in `src/blocking.py`, `src/evaluation.py`, and `src/inference.py`)*
The dataset is huge — about 2.2 million Source-1 records, 5 million
Source-2 records, and 5.3 million Source-3 records for training alone (and
a similarly large test set). Trying to process all of this in one go uses
too much memory and can crash the program. So instead, we break the
Source-1 records into batches of 50,000 at a time, and process one batch
after another. The final result is exactly the same as processing
everything at once — it's just done in smaller, safer pieces so the
pipeline runs reliably.

**How we created our validation set:** *(`src/evaluation.py`)*
Before doing any shortlisting, we split the training *entities* (not rows,
not random pairs — whole entities) into a training part and a validation
part. We do this using a fixed random seed (`42`), so the split is always
the same every time we run the code. This makes sure that all records of
one entity stay together on the same side of the split, so there's no
"cheating" by accidentally leaking information. The final model that we
use for the actual test predictions is trained again — this time on 100%
of the training data — so it learns from every example we have.

**How we measured our own performance:** *(`src/evaluation.py`)*
This matches exactly how the challenge itself is scored. For every
Source-1 entity, we check how many of its predicted matches are correct
(precision) and how many of the true matches we actually found (recall),
and combine them into an F0.5 score (this metric cares more about being
correct than about catching every single match). We then average this
score across all entities — this is called **macro entity-level F0.5**.
One important rule: if an entity genuinely has no matches and we correctly
predict "no matches", it scores a full 1.0. If we wrongly guess a match for
it, it scores 0.0.

**How we picked our final cut-off (threshold):** *(`src/evaluation.py`)*
We tried seven different cut-off points — 0.30, 0.40, 0.50, 0.60, 0.70,
0.80, and 0.90 — on our validation data, and picked whichever one gave the
best macro entity-level F0.5 score. We then used this exact same cut-off
for the test data, without any further tweaking.

**Our results on the full training/validation data**
(1,765,457 training entities / 441,364 validation entities):

| Metric | Value |
|---|---|
| Candidate recall (train) | 0.280 |
| Candidate recall (validation) | 0.280 |
| Chosen threshold | 0.90 |
| Pair-level precision at chosen threshold | 0.8286 |
| Pair-level recall at chosen threshold | 0.9577 |
| Pair-level F0.5 at chosen threshold | 0.8515 |
| **Macro entity-level F0.5 at chosen threshold (this is the official scoring metric)** | **0.3513** |
| Singleton accuracy at chosen threshold | 0.806 |

**Why is the pair-level score (0.85) so much higher than the entity-level
score (0.35)?**
This comes down to our candidate recall being only 0.280. In simple terms:
our shortlisting step only manages to include about 28% of all true matches
in its shortlist in the first place. So even if our model is very good at
picking correct matches *from the shortlist*, it can never find the other
72% of true matches — they were never even offered to it as an option. And
because we score entity-by-entity, missing even one true match for an
entity brings down that entity's own score. So the biggest opportunity to
improve our score further is to improve the shortlisting step itself (for
example, by turning on address-token shortlisting, which we currently keep
switched off for memory reasons — see Section 2).

---

## 2. Candidate Generation / Blocking Strategy — *(`src/blocking.py`)*

Checking every single Source-1 record against every Source-2 and Source-3
record is simply not possible at this scale — it would need way too much
memory and time. So instead, we use a smarter approach called **blocking**:
we use a few simple, cheap rules to quickly shortlist realistic candidates,
and we take the **union** of everything any rule finds (meaning: if even
one rule thinks two records might match, we keep that pair in our
shortlist). This favors not missing out on real matches.

| Rule | What it checks | Turned on? |
|---|---|---|
| Exact country match | Country strings match exactly after cleaning | Yes |
| Shared name words | Records share at least one meaningful word in the business name | Yes |
| Shared address words | Records share at least one meaningful word in the address | No (off by default) |
| Similar-looking name (n-grams) | Catches typos/near-duplicates that word-matching would miss | No (off by default) |
| Matching postal/PIN code | A postal code extracted from the address matches | Yes (given extra weight, since it's a strong signal) |

We build one shortlisting index from all of Source-2 + Source-3 just once
*(`build_candidate_index()` in `src/blocking.py`)*, and reuse it every time
we need it (for the training split, the validation split, and the final
run) — rebuilding it again and again would just waste time since the
underlying data doesn't change.

**Why we process the shortlisting step in batches too:** *(`generate_candidates()` in `src/blocking.py`)*
If we tried to match all Source-1 records against the shortlisting index in
a single pass, the in-between result table becomes enormous — potentially
hundreds of millions of rows — and our system ran out of memory when we
tried this at full scale. So we process Source-1 records in batches of
50,000 at a time instead. This is purely about managing memory — every
Source-1 record is still processed exactly the same way, and the final
shortlist for any entity doesn't change because of this.

**Why two of our rules are switched off by default:**
Turning on the "shared address words" and "similar-looking name" rules
would create a huge number of extra shortlist entries (tens of millions),
but only add a small improvement in recall — while using a lot more memory.
Since our two strongest rules (name words + postal code) already give
decent coverage, we keep these two extra rules off by default so the
pipeline runs smoothly and reliably. They can easily be switched back on in
our settings file *(`src/config.py`)* if someone wants to try pushing the
recall higher than our current 0.280.

Every rule that finds a candidate adds a small score to that candidate (a
postal code match counts for more, since it's a much stronger signal than
just one shared word). If a Source-1 entity ends up with more than 60
candidates after combining all rules, we only keep the top 60
highest-scoring ones, just to keep things manageable — we never throw away
a candidate that none of our rules would have picked anyway.

The file `candidate_pairs.tsv` that we submit contains exactly this final
shortlist — the same one our model actually uses to make its final
predictions — so it honestly reflects our real recall ceiling (0.280).

---

## 3. Model Architecture and Feature Engineering

### Features (how we turn a pair of records into numbers) — *(`src/features.py`)*

All our features are numbers, and we made sure none of them break or turn
into errors when a name or address is missing (a missing value is simply
treated as an empty piece of text).

- **Name-based features:** exact match, character similarity, spelling
  similarity (Levenshtein), Jaro-Winkler similarity, word overlap (Jaccard),
  number of shared words, TF-IDF cosine similarity, and the difference in
  length.
- **Address-based features:** the same set of comparisons as above, plus a
  simple yes/no flag for whether the postal codes match.
- **Other features:** whether the countries match exactly, and whether the
  candidate came from Source-2 or Source-3 (since the two sources can have
  slightly different data quality).

For the TF-IDF comparisons, we build a fresh vocabulary separately for
training and for testing, so nothing "leaks" between the two. Because
TF-IDF vectors are normalized, we're able to compute similarity for a
whole batch of pairs at once (instead of one pair at a time), which makes
things run much faster at this scale.

### Model — *(`src/model.py`)*

We used **Logistic Regression** (with `class_weight="balanced"` to handle
the fact that non-matching pairs vastly outnumber matching ones). We chose
it because:
- It gives clean probability scores, which we need for picking our
  threshold.
- It's fast enough to retrain on the entire training set.

We also kept a **Random Forest** model as an alternative option in our
settings file *(`config.MODEL_TYPE` in `src/config.py`)*, in case someone
wants to experiment with it instead.

**Keeping the training set manageable (downsampling):** *(`src/evaluation.py`
for the train/validation run, `src/inference.py` for the final full-data
training run)*
If we used every single shortlisted pair for training, we'd end up with
tens of millions of rows. So we keep *all* the true-match (positive) pairs,
but we only keep up to 10 non-matching pairs for every 1 matching pair (a
common and accepted technique for this kind of imbalanced problem). This
downsampling is only used while *training* the model — when we actually
score our validation data or the test data, we score every single pair, no
sampling involved.

**License note:** We used scikit-learn's Logistic Regression, which is
BSD-3-Clause licensed — a permissive open-source license (similar in
spirit to MIT/Apache 2.0), with no restrictions on commercial use. Neither
of our candidate models is a deep learning model, and both are far below
the 8-billion-parameter limit allowed by the challenge rules.

---

## 4. Other Relevant Information

- **Reproducibility:** Everything is controlled by one fixed random seed
  (`config.RANDOM_SEED = 42` in `src/config.py`) — the train/validation
  split, the downsampling, and the model training. All our library
  versions are pinned in `requirements.txt`. Running `python -m src.main`
  *(`src/main.py`)* on the same data will always give the same result.
- **Handling new countries:** *(`src/blocking.py` and `src/features.py`)*
  We treat "country" as just a plain text label — we never hard-code a
  fixed list of countries. This is important because the test set
  includes France, which never appears anywhere in the training data, and
  our pipeline handles it without any special changes.
- **Sanity check before submitting:** *(`src/main.py`, right before the
  output files are written)* Before writing our output files, we
  automatically check that every match we predict was actually part of
  that entity's own shortlist — if it isn't, the pipeline stops and flags
  it immediately, instead of silently submitting a broken file.
- **Fair play:** We only used the data provided under `dataset/train/` and
  `dataset/test/`. No internet lookups, external APIs, geocoding tools, or
  any government/business databases were used at any point.