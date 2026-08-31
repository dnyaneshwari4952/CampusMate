import json
import random
import re
from pathlib import Path
from typing import Optional, Union

from src.config import INTENTS_DATA_PATH
from src.data_loader import get_responses_map

FALLBACK_RESPONSES = [
    "I'm not sure I understand that. Please try asking about university courses, admission, fees, hostel, library, placements, or campus facilities.",
    "I couldn't find a matching answer for that question. Could you please rephrase it as a university-related inquiry?",
    "I don't have information on that topic in my current FAQ database. Feel free to ask about admissions, degree programs, exams, or student amenities."
]

MULTI_INTENT_RESPONSES = [
    "I noticed you asked about multiple topics in one question. I can help with one topic at a time—please ask about your questions (for example, regarding admission, fees, or courses) separately.",
    "It looks like your message combines multiple university inquiries. To give you the most accurate answer, please ask about each topic individually."
]

class ResponseManager:
    """
    Manages chatbot responses directly from data/intents.json.
    Validates response strings and supports deterministic and randomized response selection.
    """
    def __init__(self, filepath: Union[str, Path] = INTENTS_DATA_PATH, deterministic: bool = False):
        self.deterministic = deterministic
        self.filepath = Path(filepath)
        
        # Load responses from intents data
        self.responses = get_responses_map(self.filepath)
        self.responses["fallback"] = FALLBACK_RESPONSES
        self.responses["multi_intent_fallback"] = MULTI_INTENT_RESPONSES
        
        # Validate response quality
        self._validate_responses()

    def _validate_responses(self):
        """
        Scans all responses to ensure they are free of raw HTML tags, broken links,
        or developer placeholder tokens.
        """
        forbidden_patterns = [
            r'<[aA]\s+.*?>',
            r'href=["\'](LINK|ADD|YOUR)["\']',
            r'\bADD YOUR\b',
            r'\bADD LINK\b',
            r'\bXYZ\b',
            r'\bNUMBER\b',
            r'\(PLACE NAME\)',
            r'\(Place name\)',
        ]

        for intent, options in self.responses.items():
            if not options:
                raise ValueError(f"Intent '{intent}' has an empty list of responses.")
            for idx, resp in enumerate(options):
                if not isinstance(resp, str) or not resp.strip():
                    raise ValueError(f"Empty response detected in intent '{intent}' at index {idx}.")
                for pat in forbidden_patterns:
                    if re.search(pat, resp):
                        raise ValueError(
                            f"Malformed response detected in intent '{intent}' at index {idx}: '{resp}' "
                            f"(matches illegal pattern '{pat}')"
                        )

    def get_response(self, intent: str, seed: Optional[int] = None) -> str:
        """
        Retrieves an appropriate response for the predicted intent.
        
        Args:
            intent: The predicted intent tag.
            seed: Optional integer seed for reproducible random selection.
            
        Returns:
            str: Selected response string.
        """
        if intent not in self.responses or not self.responses[intent]:
            intent = "fallback"

        options = self.responses[intent]

        if self.deterministic:
            return options[0]

        if seed is not None:
            return random.Random(seed).choice(options)

        return random.choice(options)
