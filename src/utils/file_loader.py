"here i will catch the error of the json file"
from pathlib  import Path
from typing import List, Any
from src.models.function_definition import FunctionDefinition
import json
import sys

INPUTS_FOLDER = "data/input/"
def load_function_definitions(path: str) -> Any:
    try:
        full_path = Path(INPUTS_FOLDER + path)
        with open(full_path, 'r', encoding=("utf-8")) as file:
            data = json.load(file)
            return data, [FunctionDefinition.model_validate(function) for function in data]

    except  FileNotFoundError:
        print(f"ERROR: file Not found error {path}")
        sys.exit(0)
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: invalid json file: {e}")
        sys.exit(0)
