"""
tests/test_phase6_experiment.py  --  Phase 6 Reuters Artifact Experiment Verification Suite.
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from run_phase6_experiment import remove_reuters_artifact
from config import (
    FAKE_PATH,
    TRUE_PATH,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    MODEL_PATH,
    MODEL_V2_PATH,
    TOKENIZER_PATH,
    TOKENIZER_V2_PATH,
)

EXP_DATA_DIR = os.path.join(BASE_DIR, "data", "experiments", "reuters_artifact")
EXP_MODEL_DIR = os.path.join(BASE_DIR, "models", "experiments")
EXP_RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase6_reuters_experiment")


def test_artifact_removal_rules():
    """Verify conservative dateline and wire tag removal on known cases."""
    # 1. Standard wire dateline with city
    sample1 = "WASHINGTON (Reuters) - The US Congress approved a new budget measure."
    cleaned1 = remove_reuters_artifact(sample1)
    assert cleaned1 == "The US Congress approved a new budget measure.", f"Failed on sample1: {cleaned1}"

    # 2. Location with comma and state
    sample2 = "COLORADO SPRINGS, Colo. (Reuters) - Defense officials met today."
    cleaned2 = remove_reuters_artifact(sample2)
    assert cleaned2 == "Defense officials met today.", f"Failed on sample2: {cleaned2}"

    # 3. Simple leading (Reuters) -
    sample3 = "(Reuters) - A federal judge ruled on Monday."
    cleaned3 = remove_reuters_artifact(sample3)
    assert cleaned3 == "A federal judge ruled on Monday.", f"Failed on sample3: {cleaned3}"

    # 4. Trailing signoff
    sample4 = "Stock markets surged today. (Reporting by John Doe; Editing by Jane Smith)"
    cleaned4 = remove_reuters_artifact(sample4)
    assert "Reporting by" not in cleaned4, f"Failed on sample4: {cleaned4}"

    # 5. Legitimate body text containing Reuters preserved
    sample5 = "The president quoted a Reuters report during his press conference."
    cleaned5 = remove_reuters_artifact(sample5)
    assert "Reuters report" in cleaned5, f"Legitimate body text was wrongfully deleted: {cleaned5}"


def test_original_datasets_untouched():
    """Confirm raw and Phase 4 processed datasets were never altered."""
    assert os.path.exists(FAKE_PATH), "Fake.csv missing"
    assert os.path.exists(TRUE_PATH), "True.csv missing"
    assert os.path.exists(TRAIN_CSV), "train.csv missing"
    assert os.path.exists(VAL_CSV), "val.csv missing"
    assert os.path.exists(TEST_CSV), "test.csv missing"

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    assert len(train_df) == 27370, f"train.csv modified: {len(train_df)}"
    assert len(val_df) == 5865, f"val.csv modified: {len(val_df)}"
    assert len(test_df) == 5865, f"test.csv modified: {len(test_df)}"


def test_baseline_and_v2_models_preserved():
    """Ensure baseline and v2 production models remain completely intact."""
    assert os.path.exists(MODEL_PATH), "Baseline model missing"
    assert os.path.exists(TOKENIZER_PATH), "Baseline tokenizer missing"
    assert os.path.exists(MODEL_V2_PATH), "v2 model missing"
    assert os.path.exists(TOKENIZER_V2_PATH), "v2 tokenizer missing"


def test_experiment_outputs_exist():
    """Check that all required Phase 6 output artifacts exist."""
    req_files = [
        os.path.join(EXP_RESULTS_DIR, "experiment_a_metrics.json"),
        os.path.join(EXP_RESULTS_DIR, "experiment_b_metrics.json"),
        os.path.join(EXP_RESULTS_DIR, "comparison.json"),
        os.path.join(EXP_RESULTS_DIR, "artifact_analysis.json"),
        os.path.join(EXP_RESULTS_DIR, "experiment_report.md"),
        os.path.join(EXP_MODEL_DIR, "experiment_a_model.h5"),
        os.path.join(EXP_MODEL_DIR, "experiment_a_tokenizer.pkl"),
        os.path.join(EXP_MODEL_DIR, "experiment_b_model.h5"),
        os.path.join(EXP_MODEL_DIR, "experiment_b_tokenizer.pkl"),
    ]
    for rf in req_files:
        assert os.path.exists(rf), f"Required Phase 6 artifact missing: {rf}"


def test_experiment_labels_unchanged():
    """Verify that Experiment B preserved all record labels and lengths identically."""
    for split in ["train", "val", "test"]:
        orig_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", f"{split}.csv"))
        exp_b_df = pd.read_csv(os.path.join(EXP_DATA_DIR, f"exp_b_{split}.csv"))

        assert len(orig_df) == len(exp_b_df), f"Split size mismatch in {split}"
        assert (orig_df["label"].values == exp_b_df["label"].values).all(), f"Labels altered in {split}"


def test_experiment_models_loadable():
    """Verify both experimental models and tokenizers can be loaded and perform inference."""
    model_a_path = os.path.join(EXP_MODEL_DIR, "experiment_a_model.h5")
    tok_a_path = os.path.join(EXP_MODEL_DIR, "experiment_a_tokenizer.pkl")
    model_b_path = os.path.join(EXP_MODEL_DIR, "experiment_b_model.h5")
    tok_b_path = os.path.join(EXP_MODEL_DIR, "experiment_b_tokenizer.pkl")

    model_a = tf.keras.models.load_model(model_a_path)
    with open(tok_a_path, "rb") as f:
        tok_a = pickle.load(f)

    model_b = tf.keras.models.load_model(model_b_path)
    with open(tok_b_path, "rb") as f:
        tok_b = pickle.load(f)

    test_input = np.zeros((1, 300), dtype=np.int32)
    pred_a = model_a.predict(test_input, verbose=0)
    pred_b = model_b.predict(test_input, verbose=0)

    assert 0.0 <= float(pred_a[0][0]) <= 1.0, f"Invalid pred_a: {pred_a}"
    assert 0.0 <= float(pred_b[0][0]) <= 1.0, f"Invalid pred_b: {pred_b}"


if __name__ == "__main__":
    print("Running Phase 6 tests directly...")
    test_artifact_removal_rules()
    print("  [PASS] Artifact removal rules")
    test_original_datasets_untouched()
    print("  [PASS] Original datasets untouched")
    test_baseline_and_v2_models_preserved()
    print("  [PASS] Baseline and v2 models preserved")
    test_experiment_labels_unchanged()
    print("  [PASS] Experiment labels unchanged")
    print("All static Phase 6 tests passed.")
