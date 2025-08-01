import json
from pathlib import Path

def read_file(file_path: Path) -> str:
    """Reads the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

def load_json(file_path: Path) -> dict:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def save_json(data: dict, file_path: Path):
    """Saves data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved results to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")
