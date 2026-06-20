"here i will catch the error of the json file"

from pathlib import Path
from typing import Any
from src.models.function_definition import FunctionDefinition
import json
import sys

INPUTS_FOLDER = "data/input/"


def load_function_definitions(path: str) -> Any:
    """Loads and validates function definitions from a JSON file.

    Reads a JSON file from the inputs folder, parses its contents, and
    validates each entry against the FunctionDefinition schema. Exits
    the program with an error message if the file is not found, the
    JSON is malformed, or any unexpected error occurs.

    Args:
        path (str): The relative path to the JSON file, appended to
            the INPUTS_FOLDER base path.

    Returns:
        Any: A tuple of (raw_data, validated_functions) where raw_data
            is the parsed JSON list and validated_functions is a list
            of FunctionDefinition instances.
    """
    try:
        full_path = Path(INPUTS_FOLDER + path)
        with open(full_path, "r", encoding=("utf-8")) as file:
            data = json.load(file)
            return data, [
                FunctionDefinition.model_validate(function)
                for function in data
            ]

    except FileNotFoundError:
        print(f"ERROR: file Not found error {path}")
        sys.exit(0)
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: invalid json file: {e}")
        sys.exit(0)
    except Exception:
        print("ERROR: unexpected error while loading function")
        sys.exit(0)
