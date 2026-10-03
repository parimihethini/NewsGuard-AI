import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import TRUE_PATH, FAKE_PATH, MODEL_PATH, TOKENIZER_PATH
from predict import predict_text
from explain import explain_single_example

print("=======================================================")
print("PHASE 1 EVALUATION: TESTS A, B, C ON SAVED BASELINE MODEL")
print("=======================================================")

# Load 1 sample from True.csv
true_df = pd.read_csv(TRUE_PATH)
sample_real_title = str(true_df.iloc[0]["title"])
sample_real_text = str(true_df.iloc[0]["text"])
sample_real = (sample_real_title + " " + sample_real_text).strip()

# Load 1 sample from Fake.csv
fake_df = pd.read_csv(FAKE_PATH)
sample_fake_title = str(fake_df.iloc[0]["title"])
sample_fake_text = str(fake_df.iloc[0]["text"])
sample_fake = (sample_fake_title + " " + sample_fake_text).strip()

sample_invalid = "2+2=3"

print("\n--- TEST A: KNOWN REAL ARTICLE (from data/True.csv) ---")
print(f"Title: {sample_real_title}")
label_a, prob_a = predict_text(sample_real[:500])

print("\n--- TEST B: KNOWN FAKE ARTICLE (from data/Fake.csv) ---")
print(f"Title: {sample_fake_title}")
label_b, prob_b = predict_text(sample_fake[:500])

print("\n--- TEST C: PREVIOUSLY OBSERVED INVALID INPUT (2+2=3) ---")
label_c, prob_c = predict_text(sample_invalid)

print("\n--- LIME VERIFICATION ON TEST A ---")
explain_single_example(sample_real_title)

print("\n=======================================================")
print("SUMMARY OF TEST RESULTS (SAVED BASELINE MODEL):")
print(f"TEST A (REAL sample):     Label = {label_a}, P(FAKE) = {prob_a:.6f}")
print(f"TEST B (FAKE sample):     Label = {label_b}, P(FAKE) = {prob_b:.6f}")
print(f"TEST C ('2+2=3'):         Label = {label_c}, P(FAKE) = {prob_c:.6f}")
print("=======================================================")
