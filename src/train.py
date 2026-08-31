import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import (
    CLASSIFIER_PATH,
    CONFIDENCE_THRESHOLD,
    INTENTS_DATA_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from src.data_loader import get_training_samples
from src.preprocess import preprocess_text

def compute_dataset_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of the intents dataset file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def train_model():
    """
    Loads custom intents, preprocesses patterns, evaluates generalization via a stratified
    train/test split, fits the final production classifier on all curated patterns,
    and serializes model artifacts with data integrity hashing.
    """
    print(f"Loading custom intents from {INTENTS_DATA_PATH}...")
    raw_patterns, labels = get_training_samples(INTENTS_DATA_PATH)
    dataset_hash = compute_dataset_hash(INTENTS_DATA_PATH)
    
    print(f"Loaded {len(raw_patterns)} training patterns across {len(set(labels))} unique intents.")
    print(f"Dataset SHA-256: {dataset_hash[:12]}...")

    # 1. Preprocess patterns
    processed_patterns = []
    for pat in raw_patterns:
        proc = preprocess_text(pat)
        processed_patterns.append(proc)

    df = pd.DataFrame({
        "raw_pattern": raw_patterns,
        "processed_pattern": processed_patterns,
        "intent": labels
    })

    # Drop any pattern that resulted in empty tokens after preprocessing
    empty_mask = df["processed_pattern"].str.strip() == ""
    if empty_mask.any():
        print(f"Warning: Dropping {empty_mask.sum()} empty patterns post-preprocessing.")
        df = df[~empty_mask].reset_index(drop=True)

    intent_counts = df["intent"].value_counts()
    print("Intent sample distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  - {intent}: {count} patterns")

    # 2. Stratified Train-Test Split for Independent Evaluation
    min_count = intent_counts.min()
    stratify = df["intent"] if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_pattern"],
        df["intent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    print(f"\nEvaluating generalization on independent test split ({len(X_test)} test samples)...")
    eval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train_eval = eval_vectorizer.fit_transform(X_train)
    X_test_eval = eval_vectorizer.transform(X_test)

    eval_classifier = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=300,
        random_state=RANDOM_STATE
    )
    eval_classifier.fit(X_train_eval, y_train)

    y_pred = eval_classifier.predict(X_test_eval)
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON INDEPENDENT TEST SPLIT")
    print("=" * 50)
    print(f"Test Accuracy:    {accuracy * 100:.2f}%")
    print(f"Weighted F1:      {weighted_f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 50)

    # 3. Fit Final Production Model on Full Dataset
    print("\nFitting Production TF-IDF Vectorizer and Logistic Regression on full dataset...")
    prod_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_full_vec = prod_vectorizer.fit_transform(df["processed_pattern"])

    prod_classifier = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=300,
        random_state=RANDOM_STATE
    )
    prod_classifier.fit(X_full_vec, df["intent"])

    print(f"Production Vocabulary Size: {len(prod_vectorizer.vocabulary_)} features")
    print(f"Production Classes ({len(prod_classifier.classes_)}): {sorted(list(prod_classifier.classes_))}")

    # 4. Save Production Artifacts
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(prod_vectorizer, f)
    print(f"Saved Vectorizer:  {VECTORIZER_PATH}")

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(prod_classifier, f)
    print(f"Saved Classifier:  {CLASSIFIER_PATH}")

    metadata = {
        "model_name": "TF-IDF + Logistic Regression",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_hash_sha256": dataset_hash,
        "random_seed": RANDOM_STATE,
        "test_size_ratio": TEST_SIZE,
        "total_samples": len(df),
        "train_samples_count": len(X_train),
        "test_samples_count": len(X_test),
        "num_classes": len(prod_classifier.classes_),
        "classes": sorted(list(prod_classifier.classes_)),
        "vocabulary_size": len(prod_vectorizer.vocabulary_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "parameters": {
            "tfidf_ngram_range": [1, 2],
            "tfidf_sublinear_tf": True,
            "logistic_regression_C": 10.0,
            "logistic_regression_class_weight": "balanced"
        },
        "metrics": {
            "test_accuracy": float(accuracy),
            "test_weighted_f1": float(weighted_f1)
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Metadata:    {METADATA_PATH}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_model()
