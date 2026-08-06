import json
from pathlib import Path
from typing import List, Dict, Any

class GoldenDatasetLoader:
    """Helper to load and validate the evaluation golden dataset."""
    
    @staticmethod
    def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads and validates the golden evaluation dataset.
        
        Raises:
            FileNotFoundError: If dataset path is invalid.
            ValueError: If schema keys are missing.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Golden dataset must be a list of Q&A dictionaries.")
            
        required_fields = {"question", "expected_answer", "expected_context", "expected_document"}
        for idx, item in enumerate(data):
            missing = required_fields - item.keys()
            if missing:
                raise ValueError(f"Golden dataset index {idx} is missing required fields: {missing}")
                
        return data
