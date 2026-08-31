import pytest
from src.config import INTENTS_DATA_PATH
from src.data_loader import get_responses_map, get_training_samples, load_intents_data

def test_load_intents_data_success():
    data = load_intents_data(INTENTS_DATA_PATH)
    assert isinstance(data, dict)
    assert "intents" in data
    assert len(data["intents"]) == 24

def test_load_intents_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_intents_data("non_existent_path.json")

def test_get_training_samples():
    patterns, labels = get_training_samples(INTENTS_DATA_PATH)
    assert isinstance(patterns, list)
    assert isinstance(labels, list)
    assert len(patterns) == len(labels)
    assert len(patterns) >= 350
    assert "greeting" in labels
    assert "courses" in labels
    assert "fees" in labels

def test_get_responses_map():
    resp_map = get_responses_map(INTENTS_DATA_PATH)
    assert isinstance(resp_map, dict)
    assert len(resp_map) == 24
    for tag, responses in resp_map.items():
        assert len(responses) > 0
        assert all(isinstance(r, str) and len(r) > 0 for r in responses)
