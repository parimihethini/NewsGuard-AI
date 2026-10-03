"""
src/prepare_dataset.py
======================
Phase 4 — Dataset Hygiene, Deduplication, Leakage Control, and Stratified Split.

Produces:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/dataset_stats.json
    data/processed/split_metadata.json

Rules:
  1. Raw CSVs (Fake.csv, True.csv) are NEVER modified.
  2. Fake.csv -> label 1 (FAKE);  True.csv -> label 0 (REAL).
  3. Deduplication happens BEFORE splitting (prevents leakage).
  4. Tokenizer fitted on training text only (saved to tokenizer_v2.pkl).
  5. Old hybrid_model.h5 and tokenizer.pkl are NEVER overwritten.
  6. Conflicting-label records (same content, different labels) are reported
     and excluded from the clean supervised dataset.
  7. Random seed = 42 throughout.
"""

import os, sys, json, hashlib, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FAKE_PATH, TRUE_PATH,
    PROCESSED_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV,
    DATASET_STATS_JSON, SPLIT_META_JSON,
    TOKENIZER_V2_PATH,
    MAX_WORDS, MAX_LEN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_STATE,
)
from preprocess import clean_text

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    """Lowercase, collapse whitespace — for normalized-duplicate detection."""
    return " ".join(str(s).lower().split())


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _class_counts(df, label_col="label"):
    counts = df[label_col].value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------

print("=" * 65)
print("PHASE 4 — DATASET PREPARATION")
print("=" * 65)

print("\n[1] Loading raw CSVs...")
fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

raw_fake_count = len(fake_df)
raw_true_count = len(true_df)
raw_total      = raw_fake_count + raw_true_count

print(f"    Fake.csv : {raw_fake_count:,} rows")
print(f"    True.csv : {raw_true_count:,} rows")
print(f"    Total    : {raw_total:,} rows")

# ---------------------------------------------------------------------------
# Step 2 — Assign labels and combine
# ---------------------------------------------------------------------------

print("\n[2] Assigning labels (Fake=1, Real=0)...")
fake_df = fake_df.copy()
true_df = true_df.copy()
fake_df["label"] = 1
true_df["label"] = 0

# Keep only necessary columns
for df in (fake_df, true_df):
    df["title"] = df["title"].fillna("").astype(str)
    df["text"]  = df["text"].fillna("").astype(str)

df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

# ---------------------------------------------------------------------------
# Step 3 — Empty / invalid text removal
# ---------------------------------------------------------------------------

print("\n[3] Removing empty / near-empty records...")

# Headline-only policy:
# If title is non-empty but text is empty, we keep the record as headline-only
# because the model receives title + text concatenated; a valid title alone can
# still produce a meaningful cleaned sequence.  However, completely empty
# (title == "" AND text == "") records are excluded.
df["full_raw"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()

empty_both_mask = df["full_raw"].str.len() == 0
n_empty_both = empty_both_mask.sum()

# Records with empty text but valid title
text_empty_valid_title = (
    (df["text"].str.strip() == "") &
    (df["title"].str.strip().str.len() > 0)
)
n_headline_only = int(text_empty_valid_title.sum())

# Records with text too short to be meaningful (<20 chars in body) AND empty title
title_empty_text_short = (
    (df["title"].str.strip() == "") &
    (df["text"].str.strip().str.len() < 20)
)
n_title_empty_short = int(title_empty_text_short.sum())

remove_empty = empty_both_mask | title_empty_text_short
n_removed_empty = int(remove_empty.sum())

df = df[~remove_empty].copy().reset_index(drop=True)

print(f"    Empty (both title+text)          : {n_empty_both}")
print(f"    Headline-only (empty text, valid title) : {n_headline_only}  -> KEPT (headline classified)")
print(f"    Empty title + trivially short text     : {n_title_empty_short}  -> REMOVED")
print(f"    Total removed at this step              : {n_removed_empty}")
print(f"    Remaining                               : {len(df):,}")

# ---------------------------------------------------------------------------
# Step 4 — Duplicate detection and removal (BEFORE split)
# ---------------------------------------------------------------------------

print("\n[4] Duplicate detection...")

# 4A. Exact title+text duplicates (within combined dataset)
df["_key_exact"] = df["title"].str.strip() + "|||" + df["text"].str.strip()
dup_exact_mask = df.duplicated(subset=["_key_exact"], keep="first")
n_dup_exact = int(dup_exact_mask.sum())

# 4B. Normalized text duplicates (catches whitespace/case variations)
df["_key_norm"] = df["full_raw"].apply(_norm_text)
dup_norm_mask = (~dup_exact_mask) & df.duplicated(subset=["_key_norm"], keep="first")
n_dup_norm = int(dup_norm_mask.sum())

# 4C. Cross-class conflict check BEFORE removing duplicates:
# A conflict = same exact title+text appears with BOTH label 0 and label 1.
conflict_groups = (
    df.groupby("_key_exact")["label"]
    .nunique()
)
conflict_keys = conflict_groups[conflict_groups > 1].index
n_conflicts = len(conflict_keys)
conflict_mask = df["_key_exact"].isin(conflict_keys)
n_conflict_rows = int(conflict_mask.sum())

print(f"    4A Exact dup (title+text)     : {n_dup_exact}  -> REMOVED (keep first)")
print(f"    4B Normalized dup (additional): {n_dup_norm}  -> REMOVED (keep first)")
print(f"    4C Conflicting-label records  : {n_conflict_rows} rows ({n_conflicts} unique keys)")

if n_conflicts > 0:
    print("    CONFLICT DETAIL:")
    for k in list(conflict_keys)[:5]:
        rows = df[df["_key_exact"] == k][["label", "title"]].drop_duplicates()
        print(f"      key_snippet={k[:60]!r}  labels={rows['label'].tolist()}")
    print("    -> Conflicting records EXCLUDED from clean dataset.")

# Remove: exact dups + normalized dups + conflicts
remove_dup = dup_exact_mask | dup_norm_mask | conflict_mask
n_removed_dup = int(remove_dup.sum())

df_clean = df[~remove_dup].copy().reset_index(drop=True)

print(f"    Total removed at this step    : {n_removed_dup}")
print(f"    Remaining (clean)             : {len(df_clean):,}")
print(f"    Label distribution:")
print(f"      REAL (0): {(df_clean['label']==0).sum():,}")
print(f"      FAKE (1): {(df_clean['label']==1).sum():,}")

# ---------------------------------------------------------------------------
# Step 5 — Reuters artifact measurement (no action taken)
# ---------------------------------------------------------------------------

print("\n[5] Reuters artifact measurement (informational — no records removed)...")
for lbl, lname in [(0, "REAL"), (1, "FAKE")]:
    sub = df_clean[df_clean["label"] == lbl]
    r_text  = sub["text"].str.contains("Reuters", case=True, na=False).sum()
    r_title = sub["title"].str.contains("Reuters", case=True, na=False).sum()
    print(f"    {lname}: Reuters in text = {r_text}/{len(sub)} ({r_text/len(sub)*100:.1f}%),  in title = {r_title}")

print("    NOTE: Reuters tokens are intentionally preserved in the main dataset.")
print("    A controlled comparison experiment (with vs without Reuters) is deferred")
print("    to Phase 6 evaluation. The raw text is unchanged.")

# ---------------------------------------------------------------------------
# Step 6 — Stratified 3-way split (AFTER deduplication)
# ---------------------------------------------------------------------------

print("\n[6] Stratified 3-way split: 70% / 15% / 15%...")

# Convert to plain Python structures to avoid pandas/PyArrow fancy-index issues
df_clean_reset = df_clean.reset_index(drop=True)
X   = df_clean_reset["full_raw"].tolist()          # plain list of strings
y   = df_clean_reset["label"].to_numpy(dtype=int)  # numpy int array
pos = np.arange(len(y), dtype=np.int64)            # positional index

# Step 6a: carve out 15% test
pos_tv, pos_test, y_tv, y_test = train_test_split(
    pos, y,
    test_size=TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Step 6b: from remaining 85%, carve out val
val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
pos_train, pos_val, y_train, y_val = train_test_split(
    pos_tv, y_tv,
    test_size=val_relative,
    random_state=RANDOM_STATE,
    stratify=y_tv,
)

# Reconstruct DataFrames using positional iloc (int64 arrays)
df_train = df_clean_reset.iloc[pos_train.astype(int)].copy().reset_index(drop=True)
df_val   = df_clean_reset.iloc[pos_val.astype(int)].copy().reset_index(drop=True)
df_test  = df_clean_reset.iloc[pos_test.astype(int)].copy().reset_index(drop=True)

X_train = df_train["full_raw"].tolist()
X_val   = df_val["full_raw"].tolist()
X_test  = df_test["full_raw"].tolist()

print(f"    Train : {len(df_train):,}  ({len(df_train)/len(df_clean_reset)*100:.1f}%)")
print(f"    Val   : {len(df_val):,}  ({len(df_val)/len(df_clean_reset)*100:.1f}%)")
print(f"    Test  : {len(df_test):,}  ({len(df_test)/len(df_clean_reset)*100:.1f}%)")

for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    real = (split_df["label"] == 0).sum()
    fake = (split_df["label"] == 1).sum()
    print(f"    {split_name:5s}: REAL={real:,} ({real/len(split_df)*100:.1f}%)  FAKE={fake:,} ({fake/len(split_df)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Step 7 — Cross-split leakage verification (using content hash)
# ---------------------------------------------------------------------------

print("\n[7] Cross-split leakage verification...")

def _hash_list(lst):
    """Hash each string in a plain Python list."""
    return set(_sha256(_norm_text(t)) for t in lst)

train_hashes = _hash_list(X_train)
val_hashes   = _hash_list(X_val)
test_hashes  = _hash_list(X_test)

tv = train_hashes & val_hashes
tt = train_hashes & test_hashes
vt = val_hashes   & test_hashes

print(f"    Train & Val  : {len(tv)} overlapping articles")
print(f"    Train & Test : {len(tt)} overlapping articles")
print(f"    Val   & Test : {len(vt)} overlapping articles")

leakage_ok = (len(tv) == 0 and len(tt) == 0 and len(vt) == 0)
print(f"    Leakage-free : {'YES' if leakage_ok else 'NO -- REVIEW REQUIRED'}")


# ---------------------------------------------------------------------------
# Step 8 — Tokenizer: fit on training text only
# ---------------------------------------------------------------------------

print("\n[8] Fitting tokenizer on TRAINING text only...")

# Apply clean_text (same pipeline as inference) using pre-built plain lists
train_cleaned = [clean_text(t) for t in X_train]
val_cleaned   = [clean_text(t) for t in X_val]
test_cleaned  = [clean_text(t) for t in X_test]


tokenizer_v2 = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer_v2.fit_on_texts(train_cleaned)

vocab_after_train = len(tokenizer_v2.word_index)
print(f"    Vocabulary size after training fit : {vocab_after_train:,}")

# Transform (NO re-fitting)
_ = tokenizer_v2.texts_to_sequences(val_cleaned)
_ = tokenizer_v2.texts_to_sequences(test_cleaned)

vocab_after_all = len(tokenizer_v2.word_index)
print(f"    Vocabulary size after val+test transform : {vocab_after_all:,}")
tok_ok = (vocab_after_train == vocab_after_all)
print(f"    Tokenizer vocabulary unchanged by val/test: {'YES' if tok_ok else 'NO — BUG'}")

# Save tokenizer_v2 — DOES NOT overwrite tokenizer.pkl
with open(TOKENIZER_V2_PATH, "wb") as f:
    pickle.dump(tokenizer_v2, f)
print(f"    Saved: {TOKENIZER_V2_PATH}")

# ---------------------------------------------------------------------------
# Step 9 — Save split CSVs (raw text preserved — clean_text applied at model time)
# ---------------------------------------------------------------------------

print("\n[9] Saving split CSVs...")

# Drop internal helper columns before saving
drop_cols = ["_key_exact", "_key_norm", "full_raw"]
save_cols = [c for c in df_train.columns if c not in drop_cols]

df_train[save_cols].to_csv(TRAIN_CSV, index=False)
df_val[save_cols].to_csv(VAL_CSV, index=False)
df_test[save_cols].to_csv(TEST_CSV, index=False)

print(f"    {TRAIN_CSV}")
print(f"    {VAL_CSV}")
print(f"    {TEST_CSV}")

# ---------------------------------------------------------------------------
# Step 10 — Write statistics and metadata JSONs
# ---------------------------------------------------------------------------

print("\n[10] Writing metadata JSONs...")

stats = {
    "raw": {
        "fake_rows": raw_fake_count,
        "real_rows": raw_true_count,
        "total_rows": raw_total,
    },
    "removed": {
        "empty_or_trivial": n_removed_empty,
        "headline_only_kept": n_headline_only,
        "exact_duplicates": n_dup_exact,
        "normalized_duplicates": n_dup_norm,
        "conflicting_label_rows": n_conflict_rows,
        "conflicting_label_keys": n_conflicts,
        "total_removed": n_removed_empty + n_removed_dup,
    },
    "clean_total": len(df_clean),
    "clean_by_label": _class_counts(df_clean),
    "splits": {
        "train": {
            "total": len(df_train),
            "by_label": _class_counts(df_train),
        },
        "val": {
            "total": len(df_val),
            "by_label": _class_counts(df_val),
        },
        "test": {
            "total": len(df_test),
            "by_label": _class_counts(df_test),
        },
    },
    "leakage_check": {
        "train_val_overlap":  len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap":   len(vt),
        "leakage_free":       leakage_ok,
    },
    "tokenizer": {
        "fit_on": "training_text_only",
        "vocab_after_train_fit": vocab_after_train,
        "vocab_after_valtest_transform": vocab_after_all,
        "vocabulary_unchanged": tok_ok,
        "saved_path": TOKENIZER_V2_PATH,
    },
    "reuters_artifact": {
        "note": "Reuters preserved in main dataset. Controlled experiment deferred to Phase 6.",
        "real_text_contains_reuters_pct": round(
            df_clean[df_clean["label"]==0]["text"].str.contains("Reuters", case=True, na=False).mean() * 100, 2
        ),
        "fake_text_contains_reuters_pct": round(
            df_clean[df_clean["label"]==1]["text"].str.contains("Reuters", case=True, na=False).mean() * 100, 2
        ),
    },
}

with open(DATASET_STATS_JSON, "w") as f:
    json.dump(stats, f, indent=2)

meta = {
    "random_seed": RANDOM_STATE,
    "split_ratios": {
        "train": TRAIN_RATIO,
        "val":   VAL_RATIO,
        "test":  TEST_RATIO,
    },
    "deduplication_strategy": "exact_title_text + normalized_whitespace_lowercase",
    "conflict_strategy": "exclude_all_conflicting_records",
    "empty_strategy": "remove_both_empty; keep_headline_only",
    "tokenizer_strategy": "fit_on_train_only; transform_val_test",
    "raw_csvs_unchanged": True,
    "model_h5_unchanged": True,
    "tokenizer_pkl_unchanged": True,
}

with open(SPLIT_META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print(f"    {DATASET_STATS_JSON}")
print(f"    {SPLIT_META_JSON}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("PHASE 4 COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Raw total            : {raw_total:,}")
print(f"  Removed (empty)      : {n_removed_empty}")
print(f"  Removed (duplicates) : {n_dup_exact + n_dup_norm}")
print(f"  Removed (conflicts)  : {n_conflict_rows}")
print(f"  Clean total          : {len(df_clean):,}")
print(f"  Train                : {len(df_train):,}")
print(f"  Val                  : {len(df_val):,}")
print(f"  Test                 : {len(df_test):,}")
print(f"  Leakage-free splits  : {'YES' if leakage_ok else 'NO'}")
print(f"  Tokenizer vocab lock : {'YES' if tok_ok else 'NO'}")
print(f"  Seed                 : {RANDOM_STATE}")
print()
print("  Files written:")
print(f"    {TRAIN_CSV}")
print(f"    {VAL_CSV}")
print(f"    {TEST_CSV}")
print(f"    {DATASET_STATS_JSON}")
print(f"    {SPLIT_META_JSON}")
print(f"    {TOKENIZER_V2_PATH}")
print()
print("  Untouched:")
from config import FAKE_PATH, TRUE_PATH, MODEL_PATH, TOKENIZER_PATH
print(f"    {FAKE_PATH}")
print(f"    {TRUE_PATH}")
print(f"    {MODEL_PATH}")
print(f"    {TOKENIZER_PATH}")
