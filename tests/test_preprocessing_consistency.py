import os
import sys
import numpy as np

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import MODEL_PATH, TOKENIZER_PATH, MAX_LEN
from preprocess import clean_text, preprocess_for_model, load_tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from predict import predict_text
from explain import predict_proba


def test_preprocessing_consistency():
    print("=======================================================")
    print("RUNNING PHASE 1 PREPROCESSING CONSISTENCY VERIFICATION")
    print("=======================================================")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    sample_texts = [
        "Breaking: Officials report 100 new cases of virus in New York today! Visit http://example.com @user #news",
        "The Senate passed a major bipartisan spending bill following overnight debate.",
        "2+2=3",
        "Donald Trump said Reuters reported government officials signed the agreement."
    ]

    all_passed = True

    for i, raw_text in enumerate(sample_texts, 1):
        print(f"\n[Case {i}] Testing: {raw_text[:50]}...")

        # 1. Training-style preprocessing:
        cleaned_train = clean_text(raw_text)
        seq_train = tokenizer.texts_to_sequences([cleaned_train])
        pad_train = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")

        # 2. Inference preprocessing:
        pad_infer = preprocess_for_model(raw_text, tokenizer, max_len=MAX_LEN)

        # 3. Assert strict equality of token arrays
        is_equal = np.array_equal(pad_train, pad_infer)
        if is_equal:
            print(f"  -> SUCCESS: pad_train == pad_infer (Tokens: {pad_infer[pad_infer > 0][:5]}...)")
        else:
            print(f"  -> FAILED: pad_train != pad_infer")
            print(f"     pad_train: {pad_train}")
            print(f"     pad_infer: {pad_infer}")
            all_passed = False

    # 4. Verify batch consistency
    print("\n[Batch Test] Verifying multi-sample batch encoding...")
    batch_pads = preprocess_for_model(sample_texts, tokenizer, max_len=MAX_LEN)
    assert batch_pads.shape == (len(sample_texts), MAX_LEN), f"Expected shape ({len(sample_texts)}, {MAX_LEN}), got {batch_pads.shape}"
    print(f"  -> SUCCESS: Batch output shape: {batch_pads.shape}")

    # 5. Verify LIME predict_proba pipeline
    print("\n[LIME Pipeline Test] Verifying predict_proba consistency...")
    model = load_model(MODEL_PATH)
    lime_probs = predict_proba(sample_texts[:2], model, tokenizer)
    assert lime_probs.shape == (2, 2), f"Expected shape (2, 2), got {lime_probs.shape}"
    prob_sums = lime_probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0), f"Probabilities do not sum to 1.0: {prob_sums}"
    print(f"  -> SUCCESS: LIME predict_proba shape {lime_probs.shape}, row sums: {prob_sums}")

    print("\n=======================================================")
    if all_passed:
        print("ALL PREPROCESSING CONSISTENCY TESTS PASSED!")
    else:
        print("SOME PREPROCESSING CONSISTENCY TESTS FAILED!")
    print("=======================================================")
    return all_passed


if __name__ == "__main__":
    success = test_preprocessing_consistency()
    sys.exit(0 if success else 1)
