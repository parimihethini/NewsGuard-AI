import numpy as np
import pickle

from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MODEL_PATH, TOKENIZER_PATH, MAX_LEN

# Class names for LIME explanations
class_names = ["REAL", "FAKE"]

def load_tokenizer(path):
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def predict_proba(texts, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    pads = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(pads)
    # Model outputs P(fake). LIME expects shape (n_samples, n_classes)
    probs = np.hstack([1 - probs, probs])
    return probs

def explain_single_example(text):
    print("Loading model and tokenizer...")
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: predict_proba(x, model, tokenizer),
        num_features=10
    )

    print("\nOriginal Text:")
    print(text)
    print("\nTop contributing words (feature, weight):")
    for feature, weight in exp.as_list():
        print(f"{feature:20s}  {weight:.4f}")

    # If you're in Jupyter, you can do:
    # exp.show_in_notebook(text=True)

if __name__ == "__main__":
    sample_text = """The government has approved a new policy which will give every citizen
    free healthcare starting next month, according to a trusted official source."""
    explain_single_example(sample_text)
