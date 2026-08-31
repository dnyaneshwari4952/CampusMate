import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import (
    AMBIGUITY_MARGIN,
    CLASSIFIER_PATH,
    CONFIDENCE_THRESHOLD,
    INTENTS_DATA_PATH,
    LOG_PATH,
    MAX_INPUT_LENGTH,
    METADATA_PATH,
    UNMATCHED_LOG_PATH,
    VECTORIZER_PATH,
)
from src.preprocess import clean_text, download_nltk_resources, preprocess_text
from src.response_manager import ResponseManager

# Setup Logging
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CampusMateEngine")

# Specialized logger for unmatched queries
unmatched_logger = logging.getLogger("UnmatchedQueries")
unmatched_handler = logging.FileHandler(UNMATCHED_LOG_PATH, encoding="utf-8")
unmatched_handler.setFormatter(logging.Formatter("%(asctime)s | Query: %(message)s"))
unmatched_logger.addHandler(unmatched_handler)
unmatched_logger.setLevel(logging.INFO)


class Chatbot:
    """
    CampusMate University FAQ & Intent Assistant Engine.
    Combines text preprocessing, compatibility/OOD filtering, multi-intent detection,
    TF-IDF vectorization, Logistic Regression probability predictions, confidence/ambiguity gating,
    dialog memory, and predefined response mappings.
    """
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
        deterministic_responses: bool = False
    ):
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.max_input_length = MAX_INPUT_LENGTH
        self.debug_mode = False
        
        # Dialog memory
        self.previous_intent: Optional[str] = None
        
        # Load artifacts and response manager
        self._load_model()
        self.response_manager = ResponseManager(deterministic=deterministic_responses)
        download_nltk_resources()
        logger.info("CampusMate chatbot engine initialized successfully.")

    def _load_model(self):
        """Loads vectorizer, classifier, and metadata from saved_model/ directory."""
        if not VECTORIZER_PATH.exists() or not CLASSIFIER_PATH.exists():
            logger.error("Model artifacts missing. Please train the model first.")
            raise FileNotFoundError(
                "Model files (vectorizer.pkl / intent_classifier.pkl) are missing. "
                "Please run 'python -m src.train' first."
            )

        try:
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load vectorizer.pkl: {e}")
            raise RuntimeError(f"Corrupted vectorizer artifact: {e}")

        try:
            with open(CLASSIFIER_PATH, "rb") as f:
                self.classifier = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load intent_classifier.pkl: {e}")
            raise RuntimeError(f"Corrupted classifier artifact: {e}")

        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = None
        else:
            self.metadata = None

        # Verify class consistency against metadata
        if self.metadata and "classes" in self.metadata:
            saved_classes = set(self.metadata["classes"])
            model_classes = set(self.classifier.classes_)
            if saved_classes != model_classes:
                logger.warning("Mismatch between metadata classes and classifier classes.")

    def enable_debug(self):
        """Enables detailed diagnostic metadata."""
        self.debug_mode = True

    def disable_debug(self):
        """Disables diagnostic metadata."""
        self.debug_mode = False

    def reset_context(self):
        """Resets the multi-turn dialog memory."""
        self.previous_intent = None
        logger.info("Chatbot conversation context reset.")

    def _is_incompatible_query(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Checks whether the query requests out-of-scope actions such as creative writing,
        translation, coding generation, prompt injection, or unsupported domains (weather, stocks, crypto).
        """
        text_lower = text.lower().strip()

        # 1. Prompt Injection Attempts
        injection_patterns = [
            r"\bignore\s+(your\s+)?(previous\s+)?(all\s+)?instructions\b",
            r"\bforget\s+(your\s+)?(previous\s+)?(all\s+)?(university\s+)?instructions\b",
            r"\bdisregard\s+(your\s+)?(previous\s+)?instructions\b",
            r"\breveal\s+(your\s+)?(system\s+)?prompt\b",
            r"\bshow\s+(me\s+)?(your\s+)?(internal\s+)?rules\b",
            r"\bgive\s+(me\s+)?(the\s+)?source\s+code\b",
            r"\bsystem\s+prompt\b",
            r"\bjailbreak\b"
        ]
        for pat in injection_patterns:
            if re.search(pat, text_lower):
                return True, "Query contains prompt injection or system override attempt."

        # 2. Creative Requests (jokes, poems, stories, songs, riddles)
        creative_patterns = [
            r"\bjoke\b", r"\bjokes\b", r"\briddle\b", r"\briddles\b", r"\bpun\b", r"\bpuns\b",
            r"\bpoem\b", r"\bpoems\b", r"\bpoetry\b", r"\bverse\b",
            r"\bstory\b", r"\bstories\b", r"\btale\b", r"\btales\b",
            r"\bsong\b", r"\bsongs\b", r"\blyrics\b", r"\bsing\b"
        ]
        for pat in creative_patterns:
            if re.search(pat, text_lower):
                clean_pat = pat.replace(r"\b", "")
                return True, f"Query requests creative/generative action containing '{clean_pat}'."

        # 3. Translation Requests
        translation_patterns = [
            r"\btranslate\b", r"\btranslation\b",
            r"\binto\s+(french|spanish|german|hindi|chinese|japanese|russian|arabic)\b",
            r"\bin\s+(french|spanish|german|hindi|chinese|japanese|russian|arabic)\b"
        ]
        for pat in translation_patterns:
            if re.search(pat, text_lower):
                return True, "Query requests language translation."

        # 4. Out-of-Scope Domains (weather, stocks, crypto, general external topics)
        ood_domains = [
            r"\bweather\b", r"\btemperature\b", r"\bforecast\b", r"\bclimate\b",
            r"\bstock\s+price\b", r"\bstock\s+market\b", r"\bshares\s+price\b",
            r"\bbitcoin\b", r"\bcryptocurrency\b", r"\bcrypto\b", r"\bethereum\b",
            r"\brecipe\b", r"\bbake\s+a\s+cake\b", r"\bcook\b", r"\baction\s+movies\b",
            r"\bpresident\b", r"\bprime\s+minister\b", r"\broman\s+empire\b",
            r"\bcapital\s+of\s+france\b", r"\btallest\s+mountain\b", r"\bspeed\s+of\s+light\b",
            r"\bpizza\s+dough\b", r"\bbook\s+a\s+hotel\b", r"\bbook\s+me\s+a\s+hotel\b"
        ]
        for pat in ood_domains:
            if re.search(pat, text_lower):
                clean_pat = pat.replace(r"\b", "")
                return True, f"Query requests out-of-domain topic containing '{clean_pat}'."

        # 5. Coding Generation vs Academic Inquiry
        coding_actions = [
            r"\bwrite\s+(python|java|c\+\+|javascript|code|script)\b",
            r"\bgenerate\s+(python|java|code|script)\b",
            r"\bbuild\s+a\s+(website|web\s+app|application|bot)\b",
            r"\bcode\s+(for\s+me|an\s+application|a\s+website)\b",
            r"\bhack\s+the\b"
        ]
        for pat in coding_actions:
            if re.search(pat, text_lower):
                return True, "Query requests code generation or execution assistance."

        return False, None

    def _detect_multi_intent(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Detects if a user query combines two or more distinct high-level university domains
        (e.g., "What is the admission process and annual fee?", "courses and hostel").
        """
        text_lower = text.lower().strip()

        # Check for presence of connecting conjunctions
        conjunction_pat = r"\b(and|as well as|along with|plus|\&)\b"
        if not re.search(conjunction_pat, text_lower):
            return False, None

        # Distinct primary domain keyword groups
        domain_groups = {
            "courses": [r"\bcourse\b", r"\bcourses\b", r"\bprograms\b", r"\bdegree\b", r"\bdegrees\b", r"\bbranches\b"],
            "fees": [r"\bfee\b", r"\bfees\b", r"\btuition\b", r"\bfess\b"],
            "admission": [r"\badmission\b", r"\badmissions\b", r"\badmision\b", r"\baddmission\b", r"\benroll\b", r"\benrollment\b"],
            "hostel": [r"\bhostel\b", r"\bhostels\b", r"\bhostal\b", r"\bdormitory\b", r"\blodging\b"],
            "placement": [r"\bplacement\b", r"\bplacements\b", r"\brecruiters\b", r"\bpackages\b", r"\bplacment\b"],
            "scholarship": [r"\bscholarship\b", r"\bscholarships\b", r"\bscholrship\b"],
            "sports": [r"\bsports\b", r"\bgym\b", r"\bgymnasium\b"],
            "canteen": [r"\bcanteen\b", r"\bcafeteria\b"],
            "library": [r"\blibrary\b", r"\blibary\b", r"\blibrry\b"],
            "exams": [r"\bexams\b", r"\bexamination\b", r"\bexam\b"]
        }

        matched_domains = []
        for domain, patterns in domain_groups.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    matched_domains.append(domain)
                    break

        if len(matched_domains) >= 2:
            return True, f"Query combines multiple topics: {', '.join(matched_domains)}."

        return False, None

    def predict_intent(self, text: str) -> dict:
        """
        Runs input query through validation, OOD checks, multi-intent detection, preprocessing,
        typo safety, TF-IDF vectorization, Logistic Regression probability scoring, and confidence gating.

        Args:
            text: Raw input query.

        Returns:
            dict: Structured prediction details.
        """
        if not isinstance(text, str):
            text = ""

        # 1. Enforce length limit
        if len(text) > self.max_input_length:
            return {
                "predicted_intent": "fallback",
                "confidence": 1.0,
                "second_intent": None,
                "second_confidence": 0.0,
                "is_fallback": True,
                "reason": f"Input length exceeds maximum limit of {self.max_input_length} characters.",
                "processed_tokens": []
            }

        # 2. Compatibility check (creative, translation, coding generator, injection, OOD)
        incompatible, reason = self._is_incompatible_query(text)
        if incompatible:
            logger.info(f"Query '{text}' rejected: {reason} Routing to fallback.")
            return {
                "predicted_intent": "fallback",
                "confidence": 1.0,
                "second_intent": None,
                "second_confidence": 0.0,
                "is_fallback": True,
                "reason": reason,
                "processed_tokens": []
            }

        # 3. Multi-intent detection
        is_multi, multi_reason = self._detect_multi_intent(text)
        if is_multi:
            logger.info(f"Multi-intent query detected for '{text}': {multi_reason}")
            return {
                "predicted_intent": "multi_intent_fallback",
                "confidence": 1.0,
                "second_intent": None,
                "second_confidence": 0.0,
                "is_fallback": True,
                "reason": multi_reason,
                "processed_tokens": []
            }

        # 4. Clean and preprocess text
        cleaned_raw = clean_text(text)
        processed = preprocess_text(text)
        tokens = processed.split() if processed else []

        if not tokens:
            logger.info(f"Query '{text}' produced no vectorizable tokens. Routing to fallback.")
            return {
                "predicted_intent": "fallback",
                "confidence": 1.0,
                "second_intent": None,
                "second_confidence": 0.0,
                "is_fallback": True,
                "reason": "Empty or punctuation-only input.",
                "processed_tokens": []
            }

        # 5. Contextual Resolution for Ambiguous Short Queries / Follow-ups
        ambiguous_single_tokens = {
            "when", "where", "how", "who", "why", "what", "which",
            "yes", "no", "okay", "ok", "information", "details",
            "sure", "fine", "yeah"
        }
        ambiguous_phrases = {
            "what about it", "tell me about it", "tell me more about it",
            "can you tell me about it", "how much", "how much is it",
            "what about that", "tell me more", "more details"
        }
        raw_word_count = len(cleaned_raw.split())
        is_ambiguous_single = (raw_word_count == 1 and len(tokens) == 1 and tokens[0] in ambiguous_single_tokens)
        is_ambiguous_phrase = (cleaned_raw in ambiguous_phrases)

        if is_ambiguous_single or is_ambiguous_phrase:
            if not self.previous_intent:
                logger.info(f"Ambiguous query '{cleaned_raw}' received without prior context. Routing to fallback.")
                return {
                    "predicted_intent": "fallback",
                    "confidence": 1.0,
                    "second_intent": None,
                    "second_confidence": 0.0,
                    "is_fallback": True,
                    "reason": f"Ambiguous query '{cleaned_raw}' without conversational context.",
                    "processed_tokens": tokens
                }
            else:
                processed = f"{self.previous_intent} {processed}"
                tokens = processed.split()
                logger.info(f"Prepend previous intent context '{self.previous_intent}'. New query: '{processed}'")

        # 6. Typo Correction Gate with English Dictionary Protection
        vocab = self.vectorizer.vocabulary_
        import nltk
        from nltk.corpus import wordnet

        correction_targets = {
            "university", "admission", "placement", "library", "hostel",
            "document", "fee", "fees", "course", "canteen", "scholarship",
            "timing", "timetable", "syllabus", "location", "address",
            "contact", "branch", "available", "offered", "facility",
            "facilities", "event", "events", "exam", "exams", "sport", "sports"
        }

        corrected_tokens = []
        for token in tokens:
            if token in vocab or token.isdigit():
                corrected_tokens.append(token)
                continue

            # CRITICAL RULE: If token is a valid English word in WordNet, DO NOT correct it!
            # This protects words like 'hotel', 'hospital', 'france', 'replacement', 'weather', 'bitcoin'
            if wordnet.synsets(token):
                corrected_tokens.append(token)
                continue

            token_len = len(token)
            if token_len <= 3:
                corrected_tokens.append(token)
                continue

            max_allowed_dist = 1 if token_len <= 4 else 2
            best_match = None
            min_dist = float('inf')

            for target_word in correction_targets:
                if abs(len(target_word) - token_len) > max_allowed_dist:
                    continue
                dist = nltk.edit_distance(token, target_word)
                if dist < min_dist:
                    min_dist = dist
                    best_match = target_word

            if best_match and min_dist <= max_allowed_dist:
                logger.info(f"Corrected typo: '{token}' -> '{best_match}'")
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(token)

        if corrected_tokens != tokens:
            processed = " ".join(corrected_tokens)
            tokens = corrected_tokens

        # 7. Vocabulary Coverage Check (Out-of-Domain Detection)
        question_pronouns = {"what", "how", "when", "where", "who", "why", "which"}
        content_tokens = [t for t in tokens if t not in question_pronouns]
        tokens_for_coverage = content_tokens if content_tokens else tokens

        known_tokens = [t for t in tokens_for_coverage if t in vocab]
        coverage = len(known_tokens) / len(tokens_for_coverage) if tokens_for_coverage else 0.0

        if coverage < 0.35:
            logger.info(f"OOD query rejected: Vocabulary coverage is {coverage:.2f} (< 0.35). Routing to fallback.")
            return {
                "predicted_intent": "fallback",
                "confidence": 1.0,
                "second_intent": None,
                "second_confidence": 0.0,
                "is_fallback": True,
                "reason": f"Out-of-domain query. Vocabulary coverage was {coverage:.2f}.",
                "processed_tokens": tokens
            }

        # 8. TF-IDF Transformation & Logistic Regression Prediction
        vec = self.vectorizer.transform([processed])
        probabilities = self.classifier.predict_proba(vec)[0]
        classes = self.classifier.classes_

        sorted_indices = np.argsort(probabilities)[::-1]
        top_intent = classes[sorted_indices[0]]
        top_confidence = probabilities[sorted_indices[0]]

        second_intent = classes[sorted_indices[1]] if len(sorted_indices) > 1 else None
        second_confidence = probabilities[sorted_indices[1]] if len(sorted_indices) > 1 else 0.0

        # 9. Confidence Gating & Multi-Turn Context Fallback
        is_fallback = False
        fallback_reason = None

        if top_confidence < self.confidence_threshold:
            if len(tokens) <= 2 and self.previous_intent:
                contextual_query = f"{self.previous_intent} {processed}"
                logger.info(f"Low confidence ({top_confidence:.4f}) on short query. Appending context: '{contextual_query}'")
                
                context_vec = self.vectorizer.transform([contextual_query])
                context_probs = self.classifier.predict_proba(context_vec)[0]
                context_sorted = np.argsort(context_probs)[::-1]
                
                context_top_intent = classes[context_sorted[0]]
                context_top_conf = context_probs[context_sorted[0]]

                if context_top_conf >= self.confidence_threshold:
                    logger.info(f"Context resolution succeeded: {context_top_intent} ({context_top_conf:.4f})")
                    top_intent = context_top_intent
                    top_confidence = context_top_conf
                    second_intent = classes[context_sorted[1]] if len(context_sorted) > 1 else None
                    second_confidence = context_probs[context_sorted[1]] if len(context_sorted) > 1 else 0.0
                else:
                    is_fallback = True
                    fallback_reason = f"Confidence {top_confidence:.4f} is below threshold {self.confidence_threshold}."
            else:
                is_fallback = True
                fallback_reason = f"Confidence {top_confidence:.4f} is below threshold {self.confidence_threshold}."

        # 10. Ambiguity Margin Check
        if not is_fallback and (top_confidence - second_confidence) < self.ambiguity_margin:
            logger.info(f"Ambiguity detected! Margin ({top_confidence - second_confidence:.4f}) < {self.ambiguity_margin}.")
            is_fallback = True
            fallback_reason = f"Ambiguity detected between '{top_intent}' ({top_confidence:.4f}) and '{second_intent}' ({second_confidence:.4f})."

        if is_fallback:
            unmatched_logger.info(text)
            logger.info(f"Fallback triggered for query '{text}'. Reason: {fallback_reason}")

        return {
            "predicted_intent": "fallback" if is_fallback else top_intent,
            "confidence": float(top_confidence),
            "second_intent": second_intent,
            "second_confidence": float(second_confidence),
            "is_fallback": is_fallback,
            "reason": fallback_reason or "Confidence check passed.",
            "processed_tokens": tokens
        }

    def respond(self, text: str) -> dict:
        """
        Receives user input, predicts intent, fetches response, and updates dialog memory.

        Args:
            text: Raw user query.

        Returns:
            dict: Structured response packet containing reply text and diagnostic metadata.
        """
        if isinstance(text, str) and len(text) > self.max_input_length:
            return {
                "response": f"Your message is too long. Please keep your question under {self.max_input_length} characters.",
                "intent": "fallback",
                "confidence": 1.0,
                "debug_info": {
                    "input": text,
                    "processed_tokens": [],
                    "confidence": 1.0,
                    "predicted_intent": "fallback",
                    "reason": "Exceeded maximum character limit."
                }
            }

        pred = self.predict_intent(text)
        predicted_intent = pred["predicted_intent"]

        response_text = self.response_manager.get_response(predicted_intent)

        # Update dialog memory for non-fallback intents
        if not pred["is_fallback"] and predicted_intent not in ("fallback", "multi_intent_fallback"):
            self.previous_intent = predicted_intent

        debug_info = {
            "input": text,
            "processed_tokens": pred["processed_tokens"],
            "predicted_intent": predicted_intent,
            "confidence": pred["confidence"],
            "second_intent": pred["second_intent"],
            "second_confidence": pred["second_confidence"],
            "is_fallback": pred["is_fallback"],
            "reason": pred["reason"]
        }

        return {
            "response": response_text,
            "intent": predicted_intent,
            "confidence": pred["confidence"],
            "debug_info": debug_info
        }
