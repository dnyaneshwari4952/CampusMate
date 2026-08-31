import json
from pathlib import Path
from typing import Any, Dict
from src.config import METADATA_PATH

def load_model_metadata(filepath: Path = METADATA_PATH) -> Dict[str, Any]:
    """
    Loads saved model metadata and training metrics.
    
    Args:
        filepath: Path to model_metadata.json
        
    Returns:
        dict: Model metadata dictionary or empty dict if not found.
    """
    path = Path(filepath)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
