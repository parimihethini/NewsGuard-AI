import sys
import os
from tensorflow.keras.models import load_model

from config import MODEL_PATH, TOKENIZER_PATH, UNCERTAINTY_LOW, UNCERTAINTY_HIGH
from preprocess import encode_texts, load_tokenizer
from validation import validate_news_input

MODEL_LIMITATION_NOTE = (
    "This confidence represents the model's prediction based on learned textual patterns. "
    "It does not by itself verify the factual truth of the claim."
)


def predict_news(text: str, model=None, tokenizer=None, model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH):
    """
    Core prediction pipeline for NewsGuard AI.
    1. Validates input
    2. Enforces canonical preprocessing
    3. Evaluates model probabilities
    4. Applies validation-calibrated uncertainty policy
    """
    is_valid, val_msg = validate_news_input(text)
    if not is_valid:
        return {
            "valid": False,
            "status": "UNSUPPORTED",
            "label": "UNSUPPORTED",
            "message": val_msg,
            "prob_fake": None,
            "prob_real": None,
            "confidence": None,
            "note": MODEL_LIMITATION_NOTE,
        }

    if model is None:
        model = load_model(model_path)
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path)

    pads = encode_texts(text, tokenizer)
    prob_fake = float(model.predict(pads, verbose=0)[0][0])
    prob_real = 1.0 - prob_fake

    # Uncertainty decision policy
    if UNCERTAINTY_LOW <= prob_fake <= UNCERTAINTY_HIGH:
        label = "UNCERTAIN"
        confidence = max(prob_fake, prob_real)
        message = "The model does not have sufficient confidence to make a reliable classification."
    elif prob_fake > UNCERTAINTY_HIGH:
        label = "FAKE"
        confidence = prob_fake
        message = "The model identified patterns characteristic of misinformation."
    else:
        label = "REAL"
        confidence = prob_real
        message = "The model identified patterns characteristic of verified/factual reporting."

    return {
        "valid": True,
        "status": "SUCCESS",
        "label": label,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "confidence": confidence,
        "message": message,
        "note": MODEL_LIMITATION_NOTE,
    }


def predict_text(text: str):
    """CLI prediction entrypoint."""
    result = predict_news(text)
    print(f"\nInput: {text}")
    if not result["valid"]:
        print("Status: Unsupported Input")
        print(f"Details: {result['message']}")
        return result

    print(f"Prediction:        {result['label']}")
    print(f"Model Confidence:  {result['confidence'] * 100:.2f}%")
    print(f"FAKE Probability:  {result['prob_fake'] * 100:.2f}%")
    print(f"REAL Probability:  {result['prob_real'] * 100:.2f}%")
    print(f"Assessment:        {result['message']}")
    print(f"Scientific Note:   {result['note']}")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter news text: ")

    predict_text(input_text)

