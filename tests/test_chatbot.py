import pytest
from src.chatbot import Chatbot
from src.config import CONFIDENCE_THRESHOLD, MAX_INPUT_LENGTH

@pytest.fixture
def chatbot():
    return Chatbot(deterministic_responses=True)

def test_chatbot_initialization(chatbot):
    assert chatbot is not None
    assert chatbot.confidence_threshold == CONFIDENCE_THRESHOLD

def test_chatbot_normal_response(chatbot):
    res = chatbot.respond("hello")
    assert isinstance(res, dict)
    assert "response" in res
    assert "intent" in res
    assert "confidence" in res
    assert res["intent"] == "greeting"
    assert res["confidence"] >= CONFIDENCE_THRESHOLD

def test_chatbot_fallback_trigger(chatbot):
    res = chatbot.respond("what is the weather today?")
    assert res["intent"] == "fallback"
    assert res["debug_info"]["is_fallback"] is True

def test_chatbot_empty_input(chatbot):
    res = chatbot.respond("   ")
    assert res["intent"] == "fallback"
    assert res["debug_info"]["is_fallback"] is True

def test_chatbot_max_length_limit(chatbot):
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    res = chatbot.respond(long_string)
    assert res["intent"] == "fallback"
    assert "too long" in res["response"]

def test_chatbot_context_memory(chatbot):
    chatbot.reset_context()
    # Establish context
    res1 = chatbot.respond("tell me about the hostel")
    assert res1["intent"] == "hostel"
    assert chatbot.previous_intent == "hostel"

    # Follow-up ambiguous token with established context
    res2 = chatbot.respond("where?")
    assert isinstance(res2, dict)
    assert "response" in res2
    assert res2["intent"] != "fallback"
