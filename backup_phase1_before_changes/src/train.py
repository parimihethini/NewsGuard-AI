import os
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import (
    MODEL_V2_PATH,
    TOKENIZER_V2_PATH,
    EPOCHS,
    BATCH_SIZE,
    RESULTS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MAX_WORDS,
    MAX_LEN,
    EMBEDDING_DIM
)
from preprocess import load_and_prepare_data, save_tokenizer
from model import build_hybrid_model


def main():
    print("=======================================================")
    print("STARTING NEWSGUARD AI CLEAN RETRAINING PIPELINE (V2)")
    print("=======================================================")

    # 1. Load data with deduplication and stratified 3-way split
    (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer, test_raw_df = load_and_prepare_data(
        split_mode="3way", deduplicate=True
    )

    # 2. Save tokenizer fitted exclusively on training set
    save_tokenizer(tokenizer, TOKENIZER_V2_PATH)

    # 3. Build Hybrid CNN-BiLSTM model
    model = build_hybrid_model()

    # 4. Callbacks for safe training
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_V2_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 5. Train with strictly isolated validation set
    print("\nTraining Hybrid CNN-BiLSTM on Training Split...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Uncertainty calibration on validation set
    print("\nCalibrating uncertainty thresholds on Validation Split...")
    val_probs = model.predict(X_val, batch_size=128, verbose=0).ravel()
    # Check probabilities near decision boundary [0.4, 0.6]
    boundary_mask = (val_probs >= 0.40) & (val_probs <= 0.60)
    boundary_samples = int(boundary_mask.sum())
    print(f"Validation samples within [0.40, 0.60]: {boundary_samples} / {len(val_probs)} ({boundary_samples/len(val_probs)*100:.2f}%)")

    # 7. Final evaluation on UNTOUCHED held-out test split
    print("\n=======================================================")
    print("EVALUATING ON UNTOUCHED TEST SET (15% SPLIT)")
    print("=======================================================")
    test_probs = model.predict(X_test, batch_size=128, verbose=1).ravel()
    test_preds = (test_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average="binary")
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()
    clf_report = classification_report(y_test, test_preds, target_names=["REAL", "FAKE"], output_dict=True)
    clf_report_text = classification_report(y_test, test_preds, target_names=["REAL", "FAKE"])

    print(f"\nFinal Test Accuracy:  {acc * 100:.2f}%")
    print(f"Final Test Precision: {precision * 100:.2f}%")
    print(f"Final Test Recall:    {recall * 100:.2f}%")
    print(f"Final Test F1-Score:  {f1 * 100:.2f}%")
    print(f"Final Test ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix [[TN, FP], [FN, TP]]:\n{np.array(cm)}")
    print(f"\nClassification Report:\n{clf_report_text}")

    # 8. Save artifacts and configuration
    eval_results = {
        "model_name": "Hybrid_CNN_BiLSTM_v2",
        "model_path": MODEL_V2_PATH,
        "tokenizer_path": TOKENIZER_V2_PATH,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "history": {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])]
        }
    }

    results_file = os.path.join(RESULTS_DIR, "v2_metrics.json")
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    config_info = {
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "uncertainty_policy": {"low": 0.40, "high": 0.60}
    }
    with open(os.path.join(RESULTS_DIR, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    # 9. Also save model explicitly to ensure file is finalized
    model.save(MODEL_V2_PATH)
    print(f"\nRetraining complete! Model saved to: {MODEL_V2_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_V2_PATH}")
    print(f"Metrics saved to: {results_file}")


if __name__ == "__main__":
    main()

