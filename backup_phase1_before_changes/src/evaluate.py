import os
import sys
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    MODEL_V2_PATH,
    TOKENIZER_V2_PATH,
    RESULTS_DIR,
    MAX_LEN
)
from preprocess import load_and_prepare_data, load_tokenizer


def evaluate_model_pipeline(model_path: str, tokenizer_path: str, output_name: str = "baseline"):
    """
    Evaluates a model and its associated tokenizer on the untouched test split (15%).
    Never uses test data during training.
    Saves true, uncorrupted metrics to results/.
    """
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"WITH TOKENIZER:   {tokenizer_path}")
    print(f"=======================================================")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    # Load 3-way split
    (X_train_pad, y_train), (X_val_pad, y_val), (X_test_pad, y_test), train_tok, test_raw_df = load_and_prepare_data(
        split_mode="3way", deduplicate=True
    )

    print(f"\nLoading target model and tokenizer for evaluation...")
    model = load_model(model_path)
    eval_tokenizer = load_tokenizer(tokenizer_path)

    # Encode test set using the specific model's tokenizer
    test_clean_texts = test_raw_df["clean_text"].tolist()
    seqs = eval_tokenizer.texts_to_sequences(test_clean_texts)
    X_test_encoded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    print(f"Computing predictions on {len(X_test_encoded)} held-out test samples...")
    y_pred_probs = model.predict(X_test_encoded, batch_size=128, verbose=1).ravel()
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_labels, average="binary")
    roc_auc = float(roc_auc_score(y_test, y_pred_probs))
    cm = confusion_matrix(y_test, y_pred_labels).tolist()
    clf_report = classification_report(y_test, y_pred_labels, target_names=["REAL", "FAKE"], output_dict=True)
    clf_report_text = classification_report(y_test, y_pred_labels, target_names=["REAL", "FAKE"])

    results = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "test_samples": int(len(y_test)),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": clf_report
    }

    out_file = os.path.join(RESULTS_DIR, f"{output_name}_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================== RESULTS FOR {output_name.upper()} ====================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix [[TN, FP], [FN, TP]]:\n{np.array(cm)}")
    print(f"\nClassification Report:\n{clf_report_text}")
    print(f"Metrics saved to: {out_file}")

    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if target == "v2":
        evaluate_model_pipeline(MODEL_V2_PATH, TOKENIZER_V2_PATH, output_name="v2")
    else:
        evaluate_model_pipeline(MODEL_PATH, TOKENIZER_PATH, output_name="baseline")
