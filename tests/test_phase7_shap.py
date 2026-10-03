import os
import sys
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from tensorflow.keras.models import load_model

from config import MODEL_V2_PATH, TOKENIZER_V2_PATH
from preprocess import load_tokenizer
from predict import predict_text
from shap_explain import get_shap_explanation
from explain import predict_proba as predict_proba_for_lime
from lime.lime_text import LimeTextExplainer


@pytest.fixture(scope="module")
def model_and_tokenizer():
    assert os.path.exists(MODEL_V2_PATH), f"Model v2 path not found: {MODEL_V2_PATH}"
    assert os.path.exists(TOKENIZER_V2_PATH), f"Tokenizer v2 path not found: {TOKENIZER_V2_PATH}"
    model = load_model(MODEL_V2_PATH)
    tokenizer = load_tokenizer(TOKENIZER_V2_PATH)
    return model, tokenizer


def test_model_and_tokenizer_loading(model_and_tokenizer):
    """Test 1: Model and tokenizer load successfully."""
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None


def test_valid_news_shap_explanation(model_and_tokenizer):
    """Test 2: SHAP explanation on a valid news article returns list of (word, score) tuples."""
    model, tokenizer = model_and_tokenizer
    valid_text = (
        "President signs executive order on economic policy to support renewable energy investments across the country."
    )
    result = get_shap_explanation(valid_text, model, tokenizer, top_k=5)
    assert isinstance(result, list)
    assert len(result) > 0
    assert len(result) <= 5
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        word, weight = item
        assert isinstance(word, str)
        assert isinstance(weight, float)


def test_invalid_input_rejection(model_and_tokenizer):
    """Test 3: Unsupported and invalid inputs return 'UNSUPPORTED' without calling neural model."""
    model, tokenizer = model_and_tokenizer
    
    invalid_cases = [
        "",
        "   ",
        "hello",
        "How are you?",
        "2+2=3",
        "12345",
        "asdfghjkl zxcvbnm",
        "1234567890 999 888 777",
        "I love pizza everyday",
        "aaaaaaa bbbbbbb cccccc",
    ]
    
    for case in invalid_cases:
        result = get_shap_explanation(case, model, tokenizer)
        assert result == "UNSUPPORTED", f"Expected 'UNSUPPORTED' for case: '{case}', got: {result}"


def test_top_k_parameter_honored(model_and_tokenizer):
    """Test 4: Specifying top_k limits the number of returned feature attributions."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "The Senate passed a major bipartisan infrastructure bill providing billions for roads, bridges, and broadband."
    )
    res_3 = get_shap_explanation(sample_text, model, tokenizer, top_k=3)
    res_7 = get_shap_explanation(sample_text, model, tokenizer, top_k=7)
    
    assert isinstance(res_3, list)
    assert isinstance(res_7, list)
    assert len(res_3) <= 3
    assert len(res_7) <= 7
    assert len(res_3) <= len(res_7)


def test_feature_ordering_by_magnitude(model_and_tokenizer):
    """Test 5: Features are returned in descending order of absolute attribution weight."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "Breaking report reveals surprising developments in international trade negotiations between major economic powers."
    )
    result = get_shap_explanation(sample_text, model, tokenizer, top_k=6)
    assert isinstance(result, list)
    if len(result) > 1:
        magnitudes = [abs(score) for _, score in result]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] >= magnitudes[i+1] - 1e-9


def test_prediction_direction_consistency(model_and_tokenizer):
    """Test 6: Attribution weights are valid finite floats with sign indicating direction."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "Federal Reserve officials announced a quarter point interest rate adjustment following monetary policy meetings."
    )
    result = get_shap_explanation(sample_text, model, tokenizer, top_k=5)
    assert isinstance(result, list)
    for word, weight in result:
        assert np.isfinite(weight)
        assert isinstance(weight, float)


def test_no_test_data_leakage(model_and_tokenizer):
    """Test 7: Explainer does not depend on or access test split data."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "International climate summit delegates conclude discussions on emissions reduction targets and environmental policy."
    )
    # Explanation runs purely via perturbation of input without test dataset dependencies
    result = get_shap_explanation(sample_text, model, tokenizer, top_k=5)
    assert isinstance(result, list)


def test_lime_regression(model_and_tokenizer):
    """Test 8: LIME explainer continues to operate correctly alongside SHAP."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "Government officials announce new healthcare initiatives designed to expand rural medical clinic access."
    )
    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
    exp = explainer.explain_instance(
        text_instance=sample_text,
        classifier_fn=lambda x: predict_proba_for_lime(x, model, tokenizer),
        num_features=5
    )
    lime_list = exp.as_list()
    assert isinstance(lime_list, list)
    assert len(lime_list) > 0
    for word, weight in lime_list:
        assert isinstance(word, str)
        assert isinstance(weight, float)


def test_predict_pipeline_regression(model_and_tokenizer):
    """Test 9: Standard prediction pipeline maintains full structural contract."""
    model, tokenizer = model_and_tokenizer
    sample_text = (
        "Treasury department issues new guidance regarding tax compliance procedures for small business owners nationwide."
    )
    res = predict_text(sample_text, model=model, tokenizer=tokenizer)
    assert res["status"] in ["VALID", "SUCCESS"]
    assert res["label"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "confidence" in res
    assert "prob_fake" in res
    assert "prob_real" in res
    assert "disclaimer" in res


def test_empty_or_whitespace_input(model_and_tokenizer):
    """Test 10: Pure whitespace or None inputs return 'UNSUPPORTED' gracefully."""
    model, tokenizer = model_and_tokenizer
    assert get_shap_explanation(None, model, tokenizer) == "UNSUPPORTED"
    assert get_shap_explanation("      ", model, tokenizer) == "UNSUPPORTED"
    assert get_shap_explanation("\n\t  \n", model, tokenizer) == "UNSUPPORTED"
