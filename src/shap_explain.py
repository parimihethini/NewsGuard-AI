"""
SHAP Explainability Module for NewsGuard AI.
Provides token/word-level attribution using KernelExplainer on a word-presence
feature space consistent with the authoritative preprocessing and prediction pipeline.
"""

import re
import numpy as np
import shap
from config import MAX_LEN
from preprocess import preprocess_for_model, validate_input

def get_shap_explanation(text, model, tokenizer, top_k=10, nsamples=100):
    """
    Computes SHAP feature attribution values for words in the input text.
    
    Parameters:
        text (str): Input news text.
        model: Trained Keras model (predicts P(FAKE)).
        tokenizer: Fitted Keras Tokenizer.
        top_k (int): Number of top contributing features to return.
        nsamples (int): Number of perturbation samples for KernelExplainer.
        
    Returns:
        list[tuple[str, float]] | str:
            List of (word, shap_value) sorted by |shap_value| descending,
            or 'UNSUPPORTED' if input validation fails.
            
    Direction convention:
        Positive SHAP value (> 0) -> Evidence shifting prediction toward FAKE.
        Negative SHAP value (< 0) -> Evidence shifting prediction toward REAL.
    """
    # 1. Conservative Input Validation Gate
    is_valid, reason, _ = validate_input(text)
    if not is_valid:
        return "UNSUPPORTED"
    
    # 2. Extract unique ordered words/tokens from the input text
    # Clean and split into word tokens
    raw_words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)
    if not raw_words:
        return "UNSUPPORTED"
    
    # Deduplicate while preserving order
    words = list(dict.fromkeys(raw_words))
    num_words = len(words)
    
    if num_words == 0:
        return "UNSUPPORTED"

    # 3. Model prediction function over binary word indicator masks
    def f_word_mask(mask_matrix):
        """
        mask_matrix: 2D numpy array of shape (N, num_words) with values in {0, 1}.
        Returns: 1D numpy array of shape (N,) containing P(FAKE) predictions.
        """
        reconstructed_texts = []
        for row in mask_matrix:
            active_words = [words[i] for i in range(num_words) if row[i] == 1]
            reconstructed_texts.append(" ".join(active_words) if active_words else "")
            
        pads = preprocess_for_model(reconstructed_texts, tokenizer, max_len=MAX_LEN)
        probs = model.predict(pads, verbose=0)
        return probs.flatten()

    # 4. Background reference (all zeros: none of the target words present)
    bg_mask = np.zeros((1, num_words))
    
    # 5. Initialize KernelExplainer and compute SHAP values
    explainer = shap.KernelExplainer(f_word_mask, bg_mask)
    
    # For small texts, limit nsamples to at most 2^num_words + 2
    actual_samples = min(nsamples, (2 ** num_words) + 2) if num_words < 10 else nsamples
    
    # Target instance is all ones (all words present)
    target_instance = np.ones((1, num_words))
    shap_values = explainer.shap_values(target_instance, nsamples=actual_samples, silent=True)
    
    # Normalize shap_values output shape (KernelExplainer can return list or array)
    if isinstance(shap_values, list):
        vals = np.array(shap_values[0]).flatten()
    else:
        vals = np.array(shap_values).flatten()
        
    # 6. Pair words with their SHAP attribution values
    feature_attributions = []
    for word, val in zip(words, vals):
        feature_attributions.append((word, float(val)))
        
    # 7. Sort by magnitude |shap_value| descending and take top_k
    feature_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_attributions[:top_k]
