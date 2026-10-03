"""
Phase 3 Test Suite — Prediction / Probability / Confidence / Uncertainty Handling

Tests:
  A. Valid REAL dataset article  — prob_fake + prob_real ~= 1
  B. Valid FAKE dataset article  — prob_fake + prob_real ~= 1
  C. Valid short news headline   — prediction works
  D. Invalid: 2+2=3              — UNSUPPORTED, None probs
  E. Invalid: hello              — UNSUPPORTED, None probs
  F. LIME still works on valid input
  G. Confidence == prob(predicted class)
  H. label mapping: 0=REAL, 1=FAKE verified
  I. UNCERTAIN structure present, marked as provisional
  J. Phase 2 backward-compat: is_valid field still present
"""
import os, sys, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import pandas as pd
from config import TRUE_PATH, FAKE_PATH, MODEL_PATH, TOKENIZER_PATH, UNCERTAINTY_ENABLED, UNCERTAINTY_LOW, UNCERTAINTY_HIGH
from preprocess import load_tokenizer
from predict import predict_text, DISCLAIMER, LABEL_REAL, LABEL_FAKE, LABEL_UNCERTAIN, LABEL_UNSUPPORTED
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Load shared model & tokenizer once
# ---------------------------------------------------------------------------
print("Loading model and tokenizer...")
_model = load_model(MODEL_PATH)
_tok   = load_tokenizer(TOKENIZER_PATH)
print("Ready.\n")

true_df = pd.read_csv(TRUE_PATH)
fake_df = pd.read_csv(FAKE_PATH)

real_article = str(true_df.iloc[0]["title"] + " " + true_df.iloc[0]["text"])[:500]
fake_article = str(fake_df.iloc[0]["title"] + " " + fake_df.iloc[0]["text"])[:500]

all_passed = True
TOLERANCE  = 1e-6

def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASS]  {name}")
    else:
        print(f"  [FAIL]  {name}  {detail}")
        all_passed = False

# ---------------------------------------------------------------------------
# A. Valid REAL dataset article
# ---------------------------------------------------------------------------
print("=" * 60)
print("A. Valid REAL dataset article")
print("=" * 60)
r = predict_text(real_article, model=_model, tokenizer=_tok)
print(f"  label={r['label']}  prob_fake={r['prob_fake']:.6f}  prob_real={r['prob_real']:.6f}  confidence={r['confidence']:.6f}")

check("A1 status == VALID",           r["status"] == "VALID")
check("A2 is_valid == True (compat)", r["is_valid"] is True)
check("A3 prob_fake in [0,1]",        0.0 <= r["prob_fake"] <= 1.0)
check("A4 prob_real in [0,1]",        0.0 <= r["prob_real"] <= 1.0)
check("A5 prob_fake + prob_real ~= 1",abs((r["prob_fake"] + r["prob_real"]) - 1.0) < TOLERANCE,
      f"sum={r['prob_fake']+r['prob_real']}")
check("A6 label is FAKE/REAL/UNCERTAIN", r["label"] in (LABEL_FAKE, LABEL_REAL, LABEL_UNCERTAIN))
check("A7 disclaimer present",        bool(r["disclaimer"]))
check("A8 disclaimer != factual",     "factual truth" not in r["disclaimer"].lower() or "does not" in r["disclaimer"].lower())

# Verify confidence == prob of predicted class
if r["label"] == LABEL_FAKE:
    expected_conf = r["prob_fake"]
elif r["label"] == LABEL_REAL:
    expected_conf = r["prob_real"]
else:
    expected_conf = r["prob_fake"]  # UNCERTAIN uses prob_fake
check("A9 confidence == prob(predicted class)", abs(r["confidence"] - expected_conf) < TOLERANCE,
      f"confidence={r['confidence']}  expected={expected_conf}")

print()

# ---------------------------------------------------------------------------
# B. Valid FAKE dataset article
# ---------------------------------------------------------------------------
print("=" * 60)
print("B. Valid FAKE dataset article")
print("=" * 60)
r = predict_text(fake_article, model=_model, tokenizer=_tok)
print(f"  label={r['label']}  prob_fake={r['prob_fake']:.6f}  prob_real={r['prob_real']:.6f}  confidence={r['confidence']:.6f}")

check("B1 status == VALID",           r["status"] == "VALID")
check("B2 prob_fake + prob_real ~= 1",abs((r["prob_fake"] + r["prob_real"]) - 1.0) < TOLERANCE)
check("B3 label is FAKE/REAL/UNCERTAIN", r["label"] in (LABEL_FAKE, LABEL_REAL, LABEL_UNCERTAIN))
if r["label"] == LABEL_FAKE:
    expected_conf = r["prob_fake"]
elif r["label"] == LABEL_REAL:
    expected_conf = r["prob_real"]
else:
    expected_conf = r["prob_fake"]
check("B4 confidence == prob(predicted class)", abs(r["confidence"] - expected_conf) < TOLERANCE)
check("B5 label == FAKE",             r["label"] == LABEL_FAKE,
      f"(got {r['label']}, prob_fake={r['prob_fake']:.4f})")

print()

# ---------------------------------------------------------------------------
# C. Valid short news headline
# ---------------------------------------------------------------------------
print("=" * 60)
print("C. Valid short news headline")
print("=" * 60)
short_headline = "NASA announces a new lunar mission for 2026."
r = predict_text(short_headline, model=_model, tokenizer=_tok)
print(f"  label={r['label']}  prob_fake={r['prob_fake']:.6f}  confidence={r['confidence']:.6f}")

check("C1 status == VALID",           r["status"] == "VALID")
check("C2 prob_fake + prob_real ~= 1",abs((r["prob_fake"] + r["prob_real"]) - 1.0) < TOLERANCE)
check("C3 label is defined",          r["label"] in (LABEL_FAKE, LABEL_REAL, LABEL_UNCERTAIN))

print()

# ---------------------------------------------------------------------------
# D. Invalid: 2+2=3
# ---------------------------------------------------------------------------
print("=" * 60)
print("D. Invalid: '2+2=3'")
print("=" * 60)
r = predict_text("2+2=3", model=_model, tokenizer=_tok)
print(f"  label={r['label']}  prob_fake={r['prob_fake']}  prob_real={r['prob_real']}  confidence={r['confidence']}")

check("D1 status == UNSUPPORTED",     r["status"] == "UNSUPPORTED")
check("D2 label == UNSUPPORTED",      r["label"] == LABEL_UNSUPPORTED)
check("D3 prob_fake is None",         r["prob_fake"] is None)
check("D4 prob_real is None",         r["prob_real"] is None)
check("D5 confidence is None",        r["confidence"] is None)
check("D6 is_valid == False (compat)",r["is_valid"] is False)
check("D7 disclaimer present",        bool(r["disclaimer"]))

print()

# ---------------------------------------------------------------------------
# E. Invalid: hello
# ---------------------------------------------------------------------------
print("=" * 60)
print("E. Invalid: 'hello'")
print("=" * 60)
r = predict_text("hello", model=_model, tokenizer=_tok)
print(f"  label={r['label']}  prob_fake={r['prob_fake']}  prob_real={r['prob_real']}  confidence={r['confidence']}")

check("E1 status == UNSUPPORTED",     r["status"] == "UNSUPPORTED")
check("E2 prob_fake is None",         r["prob_fake"] is None)
check("E3 prob_real is None",         r["prob_real"] is None)
check("E4 confidence is None",        r["confidence"] is None)

print()

# ---------------------------------------------------------------------------
# F. LIME works for valid input
# ---------------------------------------------------------------------------
print("=" * 60)
print("F. LIME test on valid input")
print("=" * 60)
from lime.lime_text import LimeTextExplainer
from preprocess import preprocess_for_model

def predict_proba_for_lime(texts, model, tokenizer):
    pads = preprocess_for_model(texts, tokenizer, max_len=300)
    probs_fake = model.predict(pads, verbose=0)
    return np.hstack([1 - probs_fake, probs_fake])

try:
    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
    exp = explainer.explain_instance(
        text_instance=fake_article,
        classifier_fn=lambda x: predict_proba_for_lime(x, _model, _tok),
        num_features=5
    )
    lime_features = exp.as_list()
    check("F1 LIME returned features",     len(lime_features) > 0)
    check("F2 LIME features are (str,num)",all(isinstance(f, str) and isinstance(w, float)
                                              for f, w in lime_features))
    print(f"  Top LIME features: {[(f, round(w,4)) for f,w in lime_features[:3]]}")
except Exception as e:
    check("F1 LIME did not raise",         False, str(e))

print()

# ---------------------------------------------------------------------------
# G. Numerical consistency across a batch
# ---------------------------------------------------------------------------
print("=" * 60)
print("G. Numerical consistency — batch of 5 mixed inputs")
print("=" * 60)
test_texts = [
    real_article,
    fake_article,
    "Scientists discover a new treatment for Alzheimer's disease.",
    "Government officials accused of corruption in major scandal.",
    str(true_df.iloc[5]["title"] + " " + true_df.iloc[5]["text"])[:300],
]
for txt in test_texts:
    r2 = predict_text(txt, model=_model, tokenizer=_tok)
    if r2["status"] == "VALID":
        sum_ok = abs((r2["prob_fake"] + r2["prob_real"]) - 1.0) < TOLERANCE
        range_ok = 0.0 <= r2["prob_fake"] <= 1.0 and 0.0 <= r2["prob_real"] <= 1.0
        if r2["label"] == LABEL_FAKE:
            conf_ok = abs(r2["confidence"] - r2["prob_fake"]) < TOLERANCE
        elif r2["label"] == LABEL_REAL:
            conf_ok = abs(r2["confidence"] - r2["prob_real"]) < TOLERANCE
        else:
            conf_ok = abs(r2["confidence"] - r2["prob_fake"]) < TOLERANCE
        check(f"G sum~=1  conf_match  [{r2['label']:<10}]  {txt[:40]!r}", sum_ok and range_ok and conf_ok,
              f"pf={r2['prob_fake']:.4f} pr={r2['prob_real']:.4f} conf={r2['confidence']:.4f}")

print()

# ---------------------------------------------------------------------------
# H. Label mapping verification (0=REAL, 1=FAKE)
# ---------------------------------------------------------------------------
print("=" * 60)
print("H. Label mapping check")
print("=" * 60)
# prob_fake = 1.0 should be FAKE
r_fake_extreme = predict_text(fake_article, model=_model, tokenizer=_tok)
r_real_extreme = predict_text(real_article, model=_model, tokenizer=_tok)
check("H1 High prob_fake article -> FAKE or UNCERTAIN",
      r_fake_extreme["label"] in (LABEL_FAKE, LABEL_UNCERTAIN),
      f"got {r_fake_extreme['label']}, prob_fake={r_fake_extreme['prob_fake']:.4f}")
check("H2 Low  prob_fake article -> REAL or UNCERTAIN",
      r_real_extreme["label"] in (LABEL_REAL, LABEL_UNCERTAIN),
      f"got {r_real_extreme['label']}, prob_fake={r_real_extreme['prob_fake']:.4f}")

print()

# ---------------------------------------------------------------------------
# I. UNCERTAIN structure documented as provisional
# ---------------------------------------------------------------------------
print("=" * 60)
print("I. UNCERTAIN structure / provisional threshold")
print("=" * 60)
check("I1 UNCERTAINTY_ENABLED exists",  isinstance(UNCERTAINTY_ENABLED, bool))
check("I2 UNCERTAINTY_LOW < 0.5",       UNCERTAINTY_LOW < 0.5)
check("I3 UNCERTAINTY_HIGH > 0.5",      UNCERTAINTY_HIGH > 0.5)
check("I4 UNCERTAINTY_LOW < UNCERTAINTY_HIGH", UNCERTAINTY_LOW < UNCERTAINTY_HIGH)

# Synthetic edge case: check that prob just inside range -> UNCERTAIN (if enabled)
# We can't easily force the model, so test _determine_label directly
from predict import _determine_label
mid = (UNCERTAINTY_LOW + UNCERTAINTY_HIGH) / 2
if UNCERTAINTY_ENABLED:
    check("I5 _determine_label(mid) == UNCERTAIN", _determine_label(mid) == LABEL_UNCERTAIN,
          f"mid={mid}, got {_determine_label(mid)}")
    check("I6 _determine_label(0.01) == REAL",     _determine_label(0.01) == LABEL_REAL)
    check("I7 _determine_label(0.99) == FAKE",     _determine_label(0.99) == LABEL_FAKE)
else:
    print("  [SKIP] UNCERTAINTY_ENABLED=False — UNCERTAIN branch not active")

print()

# ---------------------------------------------------------------------------
# J. Phase 2 backward-compat
# ---------------------------------------------------------------------------
print("=" * 60)
print("J. Phase 2 backward-compatibility")
print("=" * 60)
r_valid   = predict_text(fake_article, model=_model, tokenizer=_tok)
r_invalid = predict_text("hello",      model=_model, tokenizer=_tok)
check("J1 valid result has is_valid=True",    r_valid["is_valid"] is True)
check("J2 invalid result has is_valid=False", r_invalid["is_valid"] is False)

print()
print("=" * 60)
print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
print("=" * 60)
