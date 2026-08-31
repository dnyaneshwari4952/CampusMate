import pytest
from src.chatbot import Chatbot

@pytest.fixture(scope="session")
def shared_chatbot():
    """Provides a shared deterministic Chatbot instance across tests for performance."""
    return Chatbot(deterministic_responses=True)
