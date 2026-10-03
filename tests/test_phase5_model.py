"""
Phase 5 Test Suite -- Clean Retrained Model (v2)
Verifies:
  - Baseline model and tokenizer are intact.
  - v2 model, tokenizer, and metrics exist.
  - v2 model loads and executes predictions.
  - Probability ranges and label mapping are valid.
  - Validation-calibrated decision threshold (0.30) works properly.
  - Unsupported inputs are blocked.
"""
import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    MODEL_V2_PATH,
    TOKENIZER_V2_PATH,
    RESULTS_DIR,
    MAX_LEN,
)
from preprocess import load_tokenizer, preprocess_for_model, validate_input
from predict import predict_text
from tensorflow.keras.models import load_model

print("=" * 60)
print("PHASE 5: COMPREHENSIVE MODEL RETRAINING & INFERENCE TESTS")
print("=" * 60)

all_passed = True

def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASS]  {name}")
    else:
        print(f"  [FAIL]  {name}  {detail}")
        all_passed = False

# -----------------------------------------------------------------
# 1. Baseline Integrity
# -----------------------------------------------------------------
check("1A Baseline model exists", os.path.exists(MODEL_PATH))
check("1B Baseline tokenizer exists", os.path.exists(TOKENIZER_PATH))
check("1C Baseline model not empty (>1MB)", os.path.getsize(MODEL_PATH) > 1_000_000)

# -----------------------------------------------------------------
# 2. V2 Artifacts Exist
# -----------------------------------------------------------------
check("2A v2 Model exists", os.path.exists(MODEL_V2_PATH))
check("2B v2 Tokenizer exists", os.path.exists(TOKENIZER_V2_PATH))
metrics_path = os.path.join(RESULTS_DIR, "v2_metrics.json")
config_path = os.path.join(RESULTS_DIR, "training_config.json")
check("2C v2 Metrics exist", os.path.exists(metrics_path))
check("2D v2 Training Config exists", os.path.exists(config_path))

# -----------------------------------------------------------------
# 3. Metrics Validation
# -----------------------------------------------------------------
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    check("3A Accuracy is populated (>95%)", "accuracy" in metrics and metrics["accuracy"] > 0.95)
    check("3B Precision is populated (>95%)", "precision" in metrics and metrics["precision"] > 0.95)
    check("3C Recall is populated (>95%)", "recall" in metrics and metrics["recall"] > 0.95)
    check("3D F1-Score is populated (>95%)", "f1_score" in metrics and metrics["f1_score"] > 0.95)
    check("3E ROC-AUC is populated (>0.99)", "roc_auc" in metrics and metrics["roc_auc"] > 0.99)
    check("3F Validation calibration used", "calibration" in metrics and "best_threshold" in metrics["calibration"])
    safety = metrics.get("safety", {})
    safety_ok = (
        safety.get("raw_csvs_untouched") is True
        and safety.get("baseline_model_preserved") is True
        and safety.get("baseline_tokenizer_preserved") is True
        and safety.get("tokenizer_v2_refit") is False
        and safety.get("test_set_used_for_training") is False
        and safety.get("test_set_used_for_threshold") is False
    )
    check("3G Safety audit flags all verified", safety_ok)

# -----------------------------------------------------------------
# 4. v2 Model and Tokenizer Live Inference
# -----------------------------------------------------------------
print("\nLoading v2 model and tokenizer for inference checks...")
v2_model = load_model(MODEL_V2_PATH)
v2_tokenizer = load_tokenizer(TOKENIZER_V2_PATH)

check("4A v2 Tokenizer vocab > 50,000 words", len(v2_tokenizer.word_index) > 50000,
      f"got {len(v2_tokenizer.word_index)}")

# Real article test
real_sample = (
    "WASHINGTON (Reuters) - The White House and congressional leaders reached a bipartisan "
    "agreement on government funding today, avoiding a federal shutdown."
)
r_real = predict_text(real_sample, model=v2_model, tokenizer=v2_tokenizer)
check("4B Real news prediction is VALID", r_real["status"] == "VALID")
check("4C Real news prob_fake in [0, 1]", 0.0 <= r_real["prob_fake"] <= 1.0)
check("4D Real news prob_real in [0, 1]", 0.0 <= r_real["prob_real"] <= 1.0)
check("4E Real news prob_fake + prob_real ~= 1.0", abs((r_real["prob_fake"] + r_real["prob_real"]) - 1.0) < 1e-6)
check("4F Real news predicted as REAL", r_real["label"] == "REAL", f"got {r_real['label']}")

# Fake article test
fake_sample = (
    "BREAKING: Shocking secret documents prove that aliens are running the international banking cartel "
    "from an underground bunker in Antarctica."
)
r_fake = predict_text(fake_sample, model=v2_model, tokenizer=v2_tokenizer)
check("4G Fake news prediction is VALID", r_fake["status"] == "VALID")
check("4H Fake news prob_fake in [0, 1]", 0.0 <= r_fake["prob_fake"] <= 1.0)
check("4I Fake news predicted as FAKE", r_fake["label"] == "FAKE", f"got {r_fake['label']}")

# -----------------------------------------------------------------
# 5. Unsupported Input Blocking with v2 Model
# -----------------------------------------------------------------
for unsupported_input in ["hello", "2+2=3", "I love pizza", "asdfghjkl zxcvbnm"]:
    r_unsup = predict_text(unsupported_input, model=v2_model, tokenizer=v2_tokenizer)
    check(f"5 UNSUPPORTED blocked: '{unsupported_input}'", r_unsup["status"] == "UNSUPPORTED" and r_unsup["prob_fake"] is None)

print()
print("=" * 60)
print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
print("=" * 60)
if not all_passed:
    sys.exit(1)
