import json
from pathlib import Path
from typing import Union
from src.config import INTENTS_DATA_PATH

def load_intents_data(filepath: Union[str, Path] = INTENTS_DATA_PATH) -> dict:
    """
    Loads and validates the intent dataset from the specified JSON file.

    Args:
        filepath: Path to the JSON intents dataset.

    Returns:
        dict: Validated dataset dictionary containing 'intents' list.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If JSON is invalid or fails schema constraints.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Intents dataset not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in intents dataset: {e}")

    if not isinstance(data, dict):
        raise ValueError("Root JSON element must be a dictionary.")

    if "intents" not in data or not isinstance(data["intents"], list):
        raise ValueError("Dataset must contain a top-level 'intents' key with a list value.")

    intents = data["intents"]
    if not intents:
        raise ValueError("The 'intents' list must not be empty.")

    seen_tags = set()
    seen_patterns = {}

    for idx, item in enumerate(intents):
        if not isinstance(item, dict):
            raise ValueError(f"Intent at index {idx} must be a dictionary.")

        tag = item.get("tag")
        if not tag or not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Intent at index {idx} has an invalid or missing 'tag'.")
        
        tag = tag.strip()
        if tag in seen_tags:
            raise ValueError(f"Duplicate intent tag '{tag}' detected at index {idx}.")
        seen_tags.add(tag)

        patterns = item.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Intent '{tag}' must contain a non-empty list of 'patterns'.")

        for p_idx, pat in enumerate(patterns):
            if not isinstance(pat, str) or not pat.strip():
                raise ValueError(f"Intent '{tag}' contains an empty or non-string pattern at index {p_idx}.")
            pat_lower = pat.strip().lower()
            if pat_lower in seen_patterns:
                other_tag = seen_patterns[pat_lower]
                if other_tag != tag:
                    raise ValueError(f"Pattern '{pat}' in intent '{tag}' collides with intent '{other_tag}'.")
            seen_patterns[pat_lower] = tag

        responses = item.get("responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"Intent '{tag}' must contain a non-empty list of 'responses'.")

        for r_idx, resp in enumerate(responses):
            if not isinstance(resp, str) or not resp.strip():
                raise ValueError(f"Intent '{tag}' contains an empty or non-string response at index {r_idx}.")

    return data

def get_training_samples(filepath: Union[str, Path] = INTENTS_DATA_PATH) -> tuple[list[str], list[str]]:
    """
    Extracts raw pattern texts and corresponding intent labels.

    Args:
        filepath: Path to intents JSON file.

    Returns:
        tuple: (list of pattern strings, list of corresponding intent label strings)
    """
    data = load_intents_data(filepath)
    patterns = []
    labels = []

    for intent in data["intents"]:
        tag = intent["tag"]
        for pat in intent["patterns"]:
            patterns.append(pat.strip())
            labels.append(tag)

    return patterns, labels

def get_responses_map(filepath: Union[str, Path] = INTENTS_DATA_PATH) -> dict[str, list[str]]:
    """
    Extracts the mapping of intent tags to their list of valid response options.

    Args:
        filepath: Path to intents JSON file.

    Returns:
        dict: Mapping of intent tag -> list of response strings.
    """
    data = load_intents_data(filepath)
    return {intent["tag"]: [r.strip() for r in intent["responses"]] for intent in data["intents"]}
