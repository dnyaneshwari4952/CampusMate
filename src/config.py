import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"
SAVED_MODEL_DIR = BASE_DIR / "saved_model"
LOG_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Path
INTENTS_DATA_PATH = DATA_DIR / "intents.json"

# Serialized Model Artifact Paths
CLASSIFIER_PATH = SAVED_MODEL_DIR / "intent_classifier.pkl"
VECTORIZER_PATH = SAVED_MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = SAVED_MODEL_DIR / "model_metadata.json"

# Logging Paths
LOG_PATH = LOG_DIR / "chatbot.log"
UNMATCHED_LOG_PATH = LOG_DIR / "unmatched_queries.log"

# Chatbot Engine Parameters
CONFIDENCE_THRESHOLD = 0.50
AMBIGUITY_MARGIN = 0.08
MAX_INPUT_LENGTH = 500

# Training Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
