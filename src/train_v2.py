"""
src/train_v2.py  --  Phase 5 Clean Model Retraining Pipeline.

Uses Phase 4 pre-split, deduplicated CSVs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Key guarantees:
  - tokenizer_v2.pkl used as-is (already fitted on training only in Phase 4)
  - No test data used during training or threshold selection
  - Validation set used ONLY for model selection + threshold calibration
  - Output: models/hybrid_model_v2.h5  (old baseline NEVER overwritten)
  - Results: results/v2_metrics.json, results/training_config.json

Label mapping (unchanged from all previous phases):
    REAL = 0,  FAKE = 1
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Resolve src/ on the path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import (
    MODEL_PATH,
    MODEL_V2_PATH,
    TOKENIZER_V2_PATH,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    RESULTS_DIR,
    MAX_LEN,
    MAX_WORDS,
    BATCH_SIZE,
    EPOCHS,
    EMBEDDING_DIM,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
)
from model import build_hybrid_model
from preprocess import clean_text, load_tokenizer

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_split_csv(path):
    """Load a Phase 4 processed split CSV and return (cleaned_texts, labels).

    Reconstructs full_text = title + ' ' + text, then applies clean_text.
    """
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("No 'label' column in %s" % path)
    df["full_text"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).str.strip()
    print("  Applying clean_text to %d records from %s ..." % (len(df), os.path.basename(path)))
    df["clean"] = df["full_text"].apply(clean_text)
    valid_mask = df["clean"].str.len() > 0
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print("  WARNING: Dropped %d rows that became empty after cleaning." % n_dropped)
        df = df[valid_mask].reset_index(drop=True)
    texts = np.array(df["clean"].tolist(), dtype=object)
    labels = np.array(df["label"].tolist(), dtype=int)
    return texts, labels


def encode(texts, tokenizer, max_len=MAX_LEN):
    """Convert cleaned texts to padded integer sequences (transform only)."""
    seqs = tokenizer.texts_to_sequences(texts.tolist())
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def calibrate_threshold(val_probs, y_val):
    """Select the best binary threshold on the VALIDATION SET only (max macro-F1 grid search).

    Returns (best_threshold, calibration_summary_dict).
    Test set is NOT used at any point here.
    """
    candidates = [round(t, 2) for t in np.arange(0.30, 0.71, 0.02)]
    best_thresh = 0.50
    best_f1 = -1.0
    results = []
    for thresh in candidates:
        preds = (val_probs >= thresh).astype(int)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_val, preds, average="macro", zero_division=0
        )
        results.append({"threshold": thresh, "macro_f1": float(f1_macro)})
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_thresh = thresh

    boundary = int(((val_probs >= 0.40) & (val_probs <= 0.60)).sum())
    print(
        "\n  Uncertainty boundary probe [0.40,0.60]: %d/%d samples (%.2f%%)"
        % (boundary, len(val_probs), boundary / len(val_probs) * 100)
    )
    print("  Best validation threshold: %.2f  (macro-F1 = %.4f)" % (best_thresh, best_f1))

    summary = {
        "best_threshold": float(best_thresh),
        "best_macro_f1_val": float(best_f1),
        "calibration_grid": results,
        "boundary_samples_0.40_0.60": boundary,
        "boundary_percent": float(boundary / len(val_probs) * 100),
        "note": "Threshold selected on validation set only. Test set was NOT used.",
    }
    return best_thresh, summary


def main():
    print("=" * 60)
    print("NEWSGUARD AI -- PHASE 5 CLEAN MODEL RETRAINING")
    print("=" * 60)

    # Safety checks
    for path, label in [
        (TRAIN_CSV, "Train CSV"),
        (VAL_CSV, "Val CSV"),
        (TEST_CSV, "Test CSV"),
        (TOKENIZER_V2_PATH, "Tokenizer v2"),
        (MODEL_PATH, "Baseline model (must exist)"),
    ]:
        if not os.path.exists(path):
            print("  [ABORT] Required file not found: %s (%s)" % (path, label))
            sys.exit(1)
        else:
            print("  [OK]  %s" % label)

    if os.path.exists(MODEL_V2_PATH):
        print("\n  [WARNING] %s already exists and will be OVERWRITTEN." % MODEL_V2_PATH)
    print()

    # Step 1: Load Phase 4 pre-split CSVs
    print("Step 1: Loading Phase 4 processed split CSVs...")
    train_texts, y_train = load_split_csv(TRAIN_CSV)
    val_texts,   y_val   = load_split_csv(VAL_CSV)
    test_texts,  y_test  = load_split_csv(TEST_CSV)
    print("\n  Train : %d samples" % len(train_texts))
    print("  Val   : %d samples" % len(val_texts))
    print("  Test  : %d samples" % len(test_texts))
    print("  Total : %d samples" % (len(train_texts) + len(val_texts) + len(test_texts)))

    # Step 2: Load tokenizer_v2 (fitted only on training in Phase 4 -- no refitting)
    print("\nStep 2: Loading tokenizer_v2 (Phase 4, train-only fit)...")
    tokenizer = load_tokenizer(TOKENIZER_V2_PATH)
    print("  Vocabulary size : %d" % len(tokenizer.word_index))
    print("  OOV token       : %r" % tokenizer.oov_token)

    # Step 3: Encode -- transform only
    print("\nStep 3: Encoding sequences (transform only)...")
    X_train = encode(train_texts, tokenizer)
    X_val   = encode(val_texts,   tokenizer)
    X_test  = encode(test_texts,  tokenizer)
    print("  X_train : %s" % str(X_train.shape))
    print("  X_val   : %s" % str(X_val.shape))
    print("  X_test  : %s" % str(X_test.shape))

    # Step 4: Build model
    print("\nStep 4: Building Hybrid CNN-BiLSTM model...")
    model = build_hybrid_model()

    # Step 5: Callbacks
    print("\nStep 5: Setting up EarlyStopping + ModelCheckpoint callbacks...")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_V2_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Step 6: Train
    print("\nStep 6: Training on Phase 4 training split...")
    print("  Max epochs : %d  |  Batch size : %d  |  Random seed : %d" % (EPOCHS, BATCH_SIZE, RANDOM_STATE))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 7: Threshold calibration on VALIDATION SET only
    print("\nStep 7: Calibrating decision threshold on VALIDATION SET...")
    val_probs = model.predict(X_val, batch_size=128, verbose=0).ravel()
    best_thresh, calibration_summary = calibrate_threshold(val_probs, y_val)

    # Step 8: Final evaluation on UNTOUCHED test set
    print("\n" + "=" * 60)
    print("Step 8: FINAL EVALUATION ON UNTOUCHED TEST SET (15%)")
    print("=" * 60)
    test_probs = model.predict(X_test, batch_size=128, verbose=1).ravel()
    test_preds = (test_probs >= best_thresh).astype(int)

    acc                       = float(accuracy_score(y_test, test_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="binary", zero_division=0
    )
    roc_auc                   = float(roc_auc_score(y_test, test_probs))
    cm                        = confusion_matrix(y_test, test_preds).tolist()
    clf_report                = classification_report(
        y_test, test_preds,
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0,
    )
    clf_report_txt = classification_report(
        y_test, test_preds, target_names=["REAL", "FAKE"], zero_division=0
    )

    print("\n  Final Test Accuracy   : %.2f%%" % (acc * 100))
    print("  Final Test Precision  : %.2f%%" % (float(precision) * 100))
    print("  Final Test Recall     : %.2f%%" % (float(recall) * 100))
    print("  Final Test F1-Score   : %.2f%%" % (float(f1) * 100))
    print("  Final Test ROC-AUC    : %.4f"   % roc_auc)
    print("  Decision Threshold    : %.2f (calibrated on val set only)" % best_thresh)
    print("\n  Confusion Matrix [[TN, FP], [FN, TP]]:\n%s" % str(np.array(cm)))
    print("\n  Classification Report:\n%s" % clf_report_txt)

    # Step 9: Save final model weights
    print("\nStep 9: Saving final model to %s ..." % MODEL_V2_PATH)
    model.save(MODEL_V2_PATH)
    print("  Model saved.")

    # Step 10: Persist metrics + config JSON
    eval_results = {
        "phase": "5",
        "model_name": "Hybrid_CNN_BiLSTM_v2",
        "model_path": MODEL_V2_PATH,
        "tokenizer_path": TOKENIZER_V2_PATH,
        "baseline_model_path": MODEL_PATH,
        "dataset_source": "Phase 4 processed CSVs (train/val/test)",
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "decision_threshold": float(best_thresh),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "history": {
            "loss":         [float(x) for x in history.history.get("loss", [])],
            "val_loss":     [float(x) for x in history.history.get("val_loss", [])],
            "accuracy":     [float(x) for x in history.history.get("accuracy", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        },
        "calibration": calibration_summary,
        "safety": {
            "raw_csvs_untouched": True,
            "baseline_model_preserved": True,
            "baseline_tokenizer_preserved": True,
            "tokenizer_v2_refit": False,
            "test_set_used_for_training": False,
            "test_set_used_for_threshold": False,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    v2_path = os.path.join(RESULTS_DIR, "v2_metrics.json")
    with open(v2_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    training_config = {
        "phase": "5",
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_STATE,
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "epochs_actual": len(history.history.get("loss", [])),
        "early_stopping": {"monitor": "val_loss", "patience": 2},
        "uncertainty_policy": {
            "decision_threshold": float(best_thresh),
            "uncertain_low": 0.40,
            "uncertain_high": 0.60,
            "calibration": "threshold selected on val set via max macro-F1 grid search",
        },
    }
    with open(os.path.join(RESULTS_DIR, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    print("\n  v2_metrics.json saved to: %s" % v2_path)
    print("  training_config.json saved.")
    print()
    print("=" * 60)
    print("RETRAINING COMPLETE")
    print("  Old baseline : %s" % MODEL_PATH)
    print("  New v2 model : %s" % MODEL_V2_PATH)
    print("  Tokenizer v2 : %s" % TOKENIZER_V2_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
