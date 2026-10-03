import sys
from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH, TOKENIZER_PATH, MAX_LEN,
    UNCERTAINTY_ENABLED, UNCERTAINTY_LOW, UNCERTAINTY_HIGH,
)
from preprocess import preprocess_for_model, load_tokenizer, validate_input


# ---------------------------------------------------------------------------
# Label mapping (verified from training):
#   model output = sigmoid probability of class 1 (FAKE)
#   0 → REAL,  1 → FAKE
# ---------------------------------------------------------------------------
LABEL_REAL      = "REAL"
LABEL_FAKE      = "FAKE"
LABEL_UNCERTAIN = "UNCERTAIN"
LABEL_UNSUPPORTED = "UNSUPPORTED"

DISCLAIMER = (
    "Model confidence reflects the classifier's learned textual patterns. "
    "It does not independently verify the factual truth of the claim."
)


def _determine_label(prob_fake: float) -> str:
    """
    Map a sigmoid probability to FAKE / REAL / UNCERTAIN.

    UNCERTAIN fires only when UNCERTAINTY_ENABLED is True AND prob_fake
    falls within the provisional interval [UNCERTAINTY_LOW, UNCERTAINTY_HIGH].

    IMPORTANT: The interval [0.40, 0.60] is PROVISIONAL and has NOT been
    statistically calibrated.  It will be replaced after Phase 5 produces a
    proper held-out validation set.  In practice this model almost never
    produces mid-range probabilities (0/400 samples in probe), so UNCERTAIN
    will rarely fire on real news text.
    """
    if UNCERTAINTY_ENABLED and UNCERTAINTY_LOW <= prob_fake <= UNCERTAINTY_HIGH:
        return LABEL_UNCERTAIN
    return LABEL_FAKE if prob_fake >= 0.5 else LABEL_REAL


def _confidence(prob_fake: float, label: str) -> float:
    """
    Return the probability associated with the predicted class.
    Confidence = prob_fake   if label is FAKE or UNCERTAIN
    Confidence = prob_real   if label is REAL
    """
    if label in (LABEL_FAKE, LABEL_UNCERTAIN):
        return prob_fake
    return 1.0 - prob_fake   # prob_real


def predict_text(text: str, model=None, tokenizer=None) -> dict:
    """
    Authoritative inference pipeline (Phase 3).

      1. validate_input  — blocks unsupported text; model NOT called if invalid.
      2. preprocess_for_model (clean_text + tokenizer + pad_sequences)
      3. CNN-BiLSTM model  →  prob_fake  →  label / confidence

    Label mapping (verified from training):
        model output ≥ 0.5  →  FAKE  (class 1)
        model output < 0.5  →  REAL  (class 0)
        [0.40, 0.60]        →  UNCERTAIN  (provisional threshold)

    Returns a dict with keys:
        status      : "VALID" | "UNSUPPORTED"
        label       : "FAKE" | "REAL" | "UNCERTAIN" | "UNSUPPORTED"
        prob_fake   : float  | None
        prob_real   : float  | None
        confidence  : float  | None   (probability of predicted class)
        reason      : str             (validation reason)
        input_type  : str             (validation category)
        disclaimer  : str             (scientific disclaimer, always present)
        is_valid    : bool            (kept for backward-compat with Phase 2)
    """
    is_valid, reason, input_type = validate_input(text)

    if not is_valid:
        return {
            "status":     "UNSUPPORTED",
            "label":      LABEL_UNSUPPORTED,
            "prob_fake":  None,
            "prob_real":  None,
            "confidence": None,
            "reason":     reason,
            "input_type": input_type,
            "disclaimer": DISCLAIMER,
            "is_valid":   False,    # backward-compat
        }

    # --- Load lazily if not provided ---
    if model is None:
        model = load_model(MODEL_PATH)
    if tokenizer is None:
        tokenizer = load_tokenizer(TOKENIZER_PATH)

    # --- Inference ---
    pads     = preprocess_for_model(text, tokenizer, max_len=MAX_LEN)
    prob_fake = float(model.predict(pads, verbose=0)[0][0])
    prob_real = 1.0 - prob_fake

    # Numerical guard — floating-point should be fine but be explicit
    assert 0.0 <= prob_fake <= 1.0, f"prob_fake out of range: {prob_fake}"
    assert abs((prob_fake + prob_real) - 1.0) < 1e-6, "prob_fake + prob_real != 1"

    label      = _determine_label(prob_fake)
    confidence = _confidence(prob_fake, label)

    return {
        "status":     "VALID",
        "label":      label,
        "prob_fake":  prob_fake,
        "prob_real":  prob_real,
        "confidence": confidence,
        "reason":     reason,
        "input_type": input_type,
        "disclaimer": DISCLAIMER,
        "is_valid":   True,         # backward-compat
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter news text: ")

    r = predict_text(input_text)

    print()
    if r["status"] == "UNSUPPORTED":
        print(f"Status    : Unsupported Input")
        print(f"Reason    : {r['reason']}")
    else:
        print(f"Status    : {r['status']}")
        print(f"Label     : {r['label']}")
        print(f"prob_fake : {r['prob_fake']:.4f}  ({r['prob_fake']*100:.2f}%)")
        print(f"prob_real : {r['prob_real']:.4f}  ({r['prob_real']*100:.2f}%)")
        print(f"Confidence: {r['confidence']:.4f}  ({r['confidence']*100:.2f}%)")
        print()
        print(f"Note: {r['disclaimer']}")
