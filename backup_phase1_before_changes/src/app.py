import sys
import os
import numpy as np
import pickle

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lime.lime_text import LimeTextExplainer

# Import config from the same folder
from config import MODEL_PATH, TOKENIZER_PATH, MAX_LEN

# -----------------------------
# Helpers to load model & tokenizer
# -----------------------------

@st.cache_resource
def load_hybrid_model():
    model = load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def encode_text(texts, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    pads = pad_sequences(
        seqs,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    return pads

# For LIME
class_names = ["REAL", "FAKE"]

def predict_proba_for_lime(texts, model, tokenizer):
    pads = encode_text(texts, tokenizer)
    probs_fake = model.predict(pads)
    # Convert shape (n,1) to (n,2) for LIME: [P(real), P(fake)]
    probs = np.hstack([1 - probs_fake, probs_fake])
    return probs


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )

    st.title("📰 Advanced Fake News Detection")
    st.write(
        "Hybrid Deep Learning (BiLSTM + CNN) with Explainable AI (LIME). "
        "Paste any news headline/article below and check if it is **FAKE** or **REAL**."
    )

    # Load model & tokenizer
    with st.spinner("Loading model and tokenizer..."):
        model = load_hybrid_model()
        tokenizer = load_tokenizer()

    # User input
    st.subheader("🔍 Enter News Text")
    default_text = (
        "Breaking: Celebrity X has been found alive on Mars, "
        "according to anonymous leaked sources."
    )
    user_text = st.text_area(
        "News article / headline:",
        value=default_text,
        height=200
    )

    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing..."):
            pads = encode_text([user_text], tokenizer)
            prob_fake = float(model.predict(pads)[0][0])
            label = "FAKE" if prob_fake >= 0.5 else "REAL"

        # Display prediction
        st.markdown("---")
        st.subheader("📌 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Label", label)
        with col2:
            st.metric("Probability (FAKE)", f"{prob_fake:.4f}")

        if label == "FAKE":
            st.error("This news is likely **FAKE**.")
        else:
            st.success("This news is likely **REAL**.")

        # Explainable AI section
        st.markdown("---")
        st.subheader("🧠 Explainable AI (LIME)")

        explain = st.checkbox("Show explanation (top contributing words)", value=True)

        if explain:
            with st.spinner("Generating explanation with LIME..."):
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(
                    text_instance=user_text,
                    classifier_fn=lambda x: predict_proba_for_lime(x, model, tokenizer),
                    num_features=10
                )

            st.write(
                "These are the top words that influenced the prediction. "
                "Positive weight → towards **FAKE**, negative → towards **REAL**."
            )

            # Show as table
            features, weights = zip(*exp.as_list())
            st.table(
                {
                    "Word / Phrase": features,
                    "Weight (towards FAKE)": [f"{w:.4f}" for w in weights],
                }
            )

            # Optional: show raw explanation HTML (commented)
            # st.components.v1.html(exp.as_html(), height=600, scrolling=True)


if __name__ == "__main__":
    main()
