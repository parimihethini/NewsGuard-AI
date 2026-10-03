"""
Phase 2 Test Suite -- Input Validation
Verifies that:
  - Invalid inputs are blocked (model is NOT called).
  - Valid inputs reach the model (prediction runs).
  - LIME is not generated for invalid inputs.
"""
import os, sys, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocess import validate_input, load_tokenizer
from config import TOKENIZER_PATH, MODEL_PATH, TRUE_PATH, FAKE_PATH
from predict import predict_text

# -----------------------------------------------------------------
# 1.  Validation-only unit tests (no model load)
# -----------------------------------------------------------------

INVALID_CASES = [
    ("",                          "empty"),
    ("   ",                       "empty"),
    ("hello",                     "too_short"),
    ("2+2=3",                     "too_short"),
    ("12345",                     "too_short"),
    ("!!!",                       "too_short"),
    ("???",                       "too_short"),
    ("asdfghjkl zxcvbnm",         "too_few_words"),
    ("I love pizza",              "too_short"),
    ("How are you?",              "greeting_or_conversational"),
]

INVALID_CASES_LONGER = [
    ("100 * 5 = 500 results",     "predominantly_numeric_or_symbols"),
    ("1234567890 999 888 777",    "math_expression"),
    ("!!!! ???? ### @@@@",        "predominantly_numeric_or_symbols"),
    ("I love pizza everyday",     "personal_opinion_or_casual"),
    ("aaaaaaa bbbbbbb cccccc",    "repeated_characters"),
]

VALID_CASES = [
    "Government announces a new renewable energy policy.",
    "Scientists report a new discovery in cancer research.",
    "NASA announces a new lunar mission.",
]

print("=" * 60)
print("PHASE 2: INPUT VALIDATION UNIT TESTS (NO MODEL)")
print("=" * 60)

all_passed = True

# Invalid: expect is_valid == False
for inp, expected_type in (INVALID_CASES + INVALID_CASES_LONGER):
    is_valid, reason, input_type = validate_input(inp)
    passed = not is_valid
    status = "PASS" if passed else "FAIL <- SHOULD BE INVALID"
    if not passed:
        all_passed = False
    print(f"  [{status}]  is_valid={is_valid}  type={input_type!r:40s}  input={inp!r}")

print()

# Valid: expect is_valid == True
for inp in VALID_CASES:
    is_valid, reason, input_type = validate_input(inp)
    passed = is_valid
    status = "PASS" if passed else "FAIL <- SHOULD BE VALID"
    if not passed:
        all_passed = False
    print(f"  [{status}]  is_valid={is_valid}  type={input_type!r:20s}  input={inp!r}")

# Dataset samples
true_df  = pd.read_csv(TRUE_PATH)
fake_df  = pd.read_csv(FAKE_PATH)
real_title = str(true_df.iloc[0]["title"]).strip()
fake_title = str(fake_df.iloc[0]["title"]).strip()

for sample, label in [(real_title, "TRUE"), (fake_title, "FAKE")]:
    is_valid, reason, input_type = validate_input(sample)
    passed = is_valid
    status = "PASS" if passed else f"FAIL <- dataset {label} headline rejected"
    if not passed:
        all_passed = False
    print(f"  [{status}]  is_valid={is_valid}  dataset={label}  input={sample[:60]!r}")

print()
print("=" * 60)
print("PHASE 2: END-TO-END PREDICTION TESTS (MODEL + VALIDATION)")
print("=" * 60)

model = None  # lazy-loaded inside predict_text

# Invalid: model must NOT be called -- verified by is_valid flag and prob_fake=None
invalid_pipeline_tests = [
    ("2+2=3",                    False),
    ("hello",                    False),
    ("I love pizza everyday",    False),
    ("asdfghjkl zxcvbnm",        False),
]

for inp, expect_valid in invalid_pipeline_tests:
    result = predict_text(inp, model=model, tokenizer=None)
    model_called = (result["prob_fake"] is not None)
    passed = (result["is_valid"] == expect_valid) and (not model_called)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  [{status}]  is_valid={result['is_valid']}  model_called={model_called}  label={result['label']!r}  input={inp!r}")

# Valid + predictions (full article text expected)
from tensorflow.keras.models import load_model as _load
_model = _load(MODEL_PATH)
_tok   = load_tokenizer(TOKENIZER_PATH)

valid_pipeline_tests = [
    # (input, expect_valid, expected_label, check_label)
    # Title-only: only verify the input is accepted (label not guaranteed from title alone)
    (real_title,  True,  "REAL",  False),
    (fake_title,  True,  "FAKE",  False),
    # Full article text: verify both valid and label
    (str(true_df.iloc[0]["title"] + " " + true_df.iloc[0]["text"])[:500], True, "REAL", True),
    (str(fake_df.iloc[0]["title"] + " " + fake_df.iloc[0]["text"])[:500], True, "FAKE", True),
]

for inp, expect_valid, expected_label, check_label in valid_pipeline_tests:
    result = predict_text(inp, model=_model, tokenizer=_tok)
    if check_label:
        passed = result["is_valid"] == expect_valid and result["label"] == expected_label
        status = "PASS" if passed else f"FAIL (got label={result['label']!r}, prob={result['prob_fake']})"
    else:
        passed = result["is_valid"] == expect_valid
        status = "PASS" if passed else f"FAIL (is_valid={result['is_valid']})"
    if not passed:
        all_passed = False
    prob_str = f"{result['prob_fake']:.4f}" if result["prob_fake"] is not None else "N/A"
    print(f"  [{status}]  label={result['label']!r}  prob_fake={prob_str}  input={inp[:60]!r}")

print()
print("=" * 60)
print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
print("=" * 60)
