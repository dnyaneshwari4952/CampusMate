import json
import pickle
import pytest
from src.chatbot import Chatbot
from src.config import CLASSIFIER_PATH, INTENTS_DATA_PATH, METADATA_PATH, VECTORIZER_PATH
from src.data_loader import load_intents_data
from src.response_manager import ResponseManager
from src.train import compute_dataset_hash

def test_model_files_exist():
    assert VECTORIZER_PATH.exists() is True
    assert CLASSIFIER_PATH.exists() is True
    assert METADATA_PATH.exists() is True

def test_vectorizer_loading():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    assert hasattr(vectorizer, "transform")
    assert hasattr(vectorizer, "vocabulary_")
    assert len(vectorizer.vocabulary_) > 400

def test_classifier_loading():
    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)
    assert hasattr(classifier, "predict")
    assert hasattr(classifier, "predict_proba")
    assert len(classifier.classes_) == 24

def test_metadata_keys():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "TF-IDF + Logistic Regression"
    assert "metrics" in metadata
    assert "classes" in metadata
    assert "dataset_hash_sha256" in metadata
    assert len(metadata["classes"]) == 24
    assert metadata["metrics"]["test_accuracy"] >= 0.70

def test_model_data_integrity():
    data = load_intents_data(INTENTS_DATA_PATH)
    current_tags = {i["tag"] for i in data["intents"]}
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    saved_classes = set(metadata["classes"])
    assert current_tags == saved_classes, "Dataset tags do not match trained model classes!"
    
    current_hash = compute_dataset_hash(INTENTS_DATA_PATH)
    assert metadata["dataset_hash_sha256"] == current_hash, "Dataset was modified after model was trained!"

def test_response_integrity_all_intents():
    rm = ResponseManager()
    assert len(rm.responses) >= 24
    for intent, options in rm.responses.items():
        assert len(options) >= 1
        for resp in options:
            assert isinstance(resp, str)
            assert len(resp.strip()) > 10
            # Ensure no HTML tags
            assert "<" not in resp and ">" not in resp
            # Ensure no developer placeholder tokens
            assert "ADD LINK" not in resp
            assert "XYZ" not in resp
            assert "PLACE NAME" not in resp

def test_missing_model_handling(monkeypatch, tmp_path):
    # Test that missing model file raises clean descriptive error
    non_existent = tmp_path / "missing.pkl"
    monkeypatch.setattr("src.chatbot.VECTORIZER_PATH", non_existent)
    with pytest.raises(FileNotFoundError, match="Model files.*missing"):
        Chatbot()
