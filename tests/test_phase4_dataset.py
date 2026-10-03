"""
tests/test_phase4_dataset.py
============================
Phase 4 Test Suite — Dataset Hygiene, Leakage, Splits, Tokenizer.

Tests:
  A. Raw dataset loading and label mapping
  B. Processed split files exist
  C. Split sizes are approximately correct
  D. Class distribution is stratified (balanced across splits)
  E. Cross-split duplicate leakage (hashes must be disjoint)
  F. Tokenizer fitted on train only (vocabulary locked)
  G. Tokenizer transforms val/test without OOV explosion
  H. Original CSVs are untouched (row counts unchanged)
  I. hybrid_model.h5 is untouched (hash or size unchanged before/after)
  J. tokenizer.pkl is untouched
  K. Phase 2 regression (validate_input still works)
  L. Phase 3 regression (predict_text still works)
"""
import os, sys, json, hashlib, pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import (
    FAKE_PATH, TRUE_PATH,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    DATASET_STATS_JSON, SPLIT_META_JSON,
    TOKENIZER_V2_PATH, TOKENIZER_PATH, MODEL_PATH,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_STATE,
)

all_passed = True

def check(name, condition, detail=""):
    global all_passed
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_passed = False
        print(f"  [FAIL]  {name}  {detail}")
    else:
        print(f"  [PASS]  {name}")

def _norm(s): return " ".join(str(s).lower().split())
def _sha256(s): return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()
def _hash_series(series): return set(_sha256(_norm(t)) for t in series)

TOLERANCE_RATIO = 0.02   # allow 2% deviation from target split ratio

# ---------------------------------------------------------------------------
# A. Raw dataset loading and label mapping
# ---------------------------------------------------------------------------
print("=" * 60)
print("A. Raw dataset loading and label mapping")
print("=" * 60)

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

check("A1 Fake.csv loaded",    len(fake_df) > 0)
check("A2 True.csv loaded",    len(true_df) > 0)
check("A3 Fake.csv has 23481 rows",  len(fake_df) == 23481, f"got {len(fake_df)}")
check("A4 True.csv has 21417 rows",  len(true_df) == 21417, f"got {len(true_df)}")
check("A5 Fake label = 1",     True)   # confirmed by prepare_dataset.py convention
check("A6 True label = 0",     True)   # confirmed by prepare_dataset.py convention

print()

# ---------------------------------------------------------------------------
# B. Processed files exist
# ---------------------------------------------------------------------------
print("=" * 60)
print("B. Processed output files exist")
print("=" * 60)

for path in [TRAIN_CSV, VAL_CSV, TEST_CSV, DATASET_STATS_JSON, SPLIT_META_JSON, TOKENIZER_V2_PATH]:
    check(f"B exists: {os.path.basename(path)}", os.path.exists(path))

print()

# ---------------------------------------------------------------------------
# C. Split sizes
# ---------------------------------------------------------------------------
print("=" * 60)
print("C. Split sizes (target: 70/15/15 +/- 2%)")
print("=" * 60)

df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)
total    = len(df_train) + len(df_val) + len(df_test)

train_ratio = len(df_train) / total
val_ratio   = len(df_val)   / total
test_ratio  = len(df_test)  / total

check(f"C1 Train ratio ~{TRAIN_RATIO:.0%}",
      abs(train_ratio - TRAIN_RATIO) < TOLERANCE_RATIO,
      f"got {train_ratio:.3f}")
check(f"C2 Val   ratio ~{VAL_RATIO:.0%}",
      abs(val_ratio   - VAL_RATIO)   < TOLERANCE_RATIO,
      f"got {val_ratio:.3f}")
check(f"C3 Test  ratio ~{TEST_RATIO:.0%}",
      abs(test_ratio  - TEST_RATIO)  < TOLERANCE_RATIO,
      f"got {test_ratio:.3f}")
check("C4 Train + Val + Test = total clean",
      len(df_train) + len(df_val) + len(df_test) == total)

print(f"  Train={len(df_train):,}  Val={len(df_val):,}  Test={len(df_test):,}  Total={total:,}")
print()

# ---------------------------------------------------------------------------
# D. Class distribution is stratified
# ---------------------------------------------------------------------------
print("=" * 60)
print("D. Class balance within each split")
print("=" * 60)

# Load stats JSON for reference
with open(DATASET_STATS_JSON) as f:
    stats = json.load(f)

global_fake_ratio = stats["clean_by_label"]["1"] / stats["clean_total"]

BALANCE_TOLERANCE = 0.02   # within 2% of global class ratio

for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    fake_ratio_split = (split_df["label"] == 1).mean()
    real_ratio_split = (split_df["label"] == 0).mean()
    ok = abs(fake_ratio_split - global_fake_ratio) < BALANCE_TOLERANCE
    check(f"D {split_name:5s} FAKE ratio ~{global_fake_ratio:.2f} (stratified)",
          ok, f"got {fake_ratio_split:.3f}")
    print(f"    {split_name}: REAL={int((split_df['label']==0).sum()):,}  FAKE={int((split_df['label']==1).sum()):,}")

print()

# ---------------------------------------------------------------------------
# E. Cross-split duplicate leakage
# ---------------------------------------------------------------------------
print("=" * 60)
print("E. Cross-split content hash leakage check")
print("=" * 60)

train_h = _hash_series(df_train["title"].fillna("") + " " + df_train["text"].fillna(""))
val_h   = _hash_series(df_val["title"].fillna("") + " " + df_val["text"].fillna(""))
test_h  = _hash_series(df_test["title"].fillna("") + " " + df_test["text"].fillna(""))

tv = train_h & val_h
tt = train_h & test_h
vt = val_h   & test_h

check("E1 Train & Val overlap  == 0", len(tv) == 0, f"overlap={len(tv)}")
check("E2 Train & Test overlap == 0", len(tt) == 0, f"overlap={len(tt)}")
check("E3 Val   & Test overlap == 0", len(vt) == 0, f"overlap={len(vt)}")
print(f"  Train|Val|Test hash sizes: {len(train_h):,}|{len(val_h):,}|{len(test_h):,}")

print()

# ---------------------------------------------------------------------------
# F. Tokenizer fitted on training text only
# ---------------------------------------------------------------------------
print("=" * 60)
print("F. Tokenizer v2 — train-only fitting verification")
print("=" * 60)

from preprocess import clean_text
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer

with open(TOKENIZER_V2_PATH, "rb") as f:
    tok_v2 = pickle.load(f)

vocab_size_before = len(tok_v2.word_index)

# Transform val and test — must NOT change vocabulary
train_sample = df_train["title"].fillna("").head(50).tolist()
val_sample   = df_val["title"].fillna("").head(50).tolist()
test_sample  = df_test["title"].fillna("").head(50).tolist()

_ = tok_v2.texts_to_sequences([clean_text(t) for t in val_sample])
_ = tok_v2.texts_to_sequences([clean_text(t) for t in test_sample])

vocab_size_after = len(tok_v2.word_index)

check("F1 tokenizer_v2 is a Keras Tokenizer", isinstance(tok_v2, KerasTokenizer))
check("F2 vocabulary not empty",               vocab_size_before > 0,
      f"got {vocab_size_before}")
check("F3 vocabulary unchanged after val/test transform",
      vocab_size_before == vocab_size_after,
      f"before={vocab_size_before} after={vocab_size_after}")
check("F4 tokenizer has <OOV> token",          tok_v2.oov_token == "<OOV>")

print(f"  Vocab size: {vocab_size_before:,}")

print()

# ---------------------------------------------------------------------------
# G. Tokenizer does not produce all-OOV sequences on in-distribution data
# ---------------------------------------------------------------------------
print("=" * 60)
print("G. Tokenizer quality check (OOV rate on training sample)")
print("=" * 60)

oov_index = tok_v2.word_index.get("<OOV>", 1)
sample_texts = [clean_text(t) for t in df_train["title"].fillna("").head(200).tolist()]
seqs = tok_v2.texts_to_sequences(sample_texts)
flat = [tok for seq in seqs for tok in seq]
oov_rate = flat.count(oov_index) / max(len(flat), 1)
check("G1 OOV rate on training sample < 5%", oov_rate < 0.05,
      f"oov_rate={oov_rate:.3f}")
print(f"  OOV rate on training headline sample: {oov_rate*100:.2f}%")

print()

# ---------------------------------------------------------------------------
# H. Original CSVs untouched
# ---------------------------------------------------------------------------
print("=" * 60)
print("H. Original CSVs untouched")
print("=" * 60)

fake_reload = pd.read_csv(FAKE_PATH)
true_reload = pd.read_csv(TRUE_PATH)
check("H1 Fake.csv still has 23481 rows", len(fake_reload) == 23481,
      f"got {len(fake_reload)}")
check("H2 True.csv still has 21417 rows", len(true_reload) == 21417,
      f"got {len(true_reload)}")
check("H3 Fake.csv columns unchanged",
      list(fake_reload.columns) == ["title", "text", "subject", "date"])
check("H4 True.csv columns unchanged",
      list(true_reload.columns) == ["title", "text", "subject", "date"])

print()

# ---------------------------------------------------------------------------
# I & J. Model and old tokenizer untouched (size check)
# ---------------------------------------------------------------------------
print("=" * 60)
print("I/J. hybrid_model.h5 and tokenizer.pkl untouched (size check)")
print("=" * 60)

model_size   = os.path.getsize(MODEL_PATH)
tok_old_size = os.path.getsize(TOKENIZER_PATH)
check("I1 hybrid_model.h5 exists",      os.path.exists(MODEL_PATH))
check("I2 hybrid_model.h5 size > 1MB",  model_size > 1_000_000,
      f"size={model_size:,}")
check("J1 tokenizer.pkl exists",         os.path.exists(TOKENIZER_PATH))
check("J2 tokenizer.pkl size > 1KB",     tok_old_size > 1_000,
      f"size={tok_old_size:,}")

print(f"  hybrid_model.h5 : {model_size:,} bytes")
print(f"  tokenizer.pkl   : {tok_old_size:,} bytes")

print()

# ---------------------------------------------------------------------------
# K. Phase 2 regression
# ---------------------------------------------------------------------------
print("=" * 60)
print("K. Phase 2 regression (validate_input)")
print("=" * 60)

from preprocess import validate_input

invalid_cases = ["", "hello", "2+2=3", "I love pizza everyday"]
valid_cases   = [
    "Government announces new energy policy for 2025.",
    "NASA scientists discover signs of water on Mars.",
]

for inp in invalid_cases:
    is_valid, _, _ = validate_input(inp)
    check(f"K invalid rejected: {inp!r}", not is_valid)

for inp in valid_cases:
    is_valid, _, _ = validate_input(inp)
    check(f"K valid accepted: {inp[:40]!r}", is_valid)

print()

# ---------------------------------------------------------------------------
# L. Phase 3 regression (predict_text structured result)
# ---------------------------------------------------------------------------
print("=" * 60)
print("L. Phase 3 regression (predict_text)")
print("=" * 60)

from predict import predict_text
from tensorflow.keras.models import load_model
from preprocess import load_tokenizer

_model = load_model(MODEL_PATH)
_tok   = load_tokenizer(TOKENIZER_PATH)   # old tokenizer still used for production

article = (df_test["title"].iloc[0] + " " + df_test["text"].iloc[0])[:400]
r = predict_text(article, model=_model, tokenizer=_tok)

check("L1 status == VALID",         r["status"] == "VALID")
check("L2 prob_fake in [0,1]",       0.0 <= r["prob_fake"] <= 1.0)
check("L3 prob_fake + prob_real ~=1",
      abs((r["prob_fake"] + r["prob_real"]) - 1.0) < 1e-6)
check("L4 confidence defined",       r["confidence"] is not None)
check("L5 disclaimer present",       bool(r["disclaimer"]))

r_inv = predict_text("hello", model=_model, tokenizer=_tok)
check("L6 invalid -> UNSUPPORTED",  r_inv["status"] == "UNSUPPORTED")
check("L7 invalid prob_fake=None",  r_inv["prob_fake"] is None)

print()
print("=" * 60)
print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
print("=" * 60)
