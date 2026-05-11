"here i will catch the error of the json file"
from pathlib  import Path
from src.models.function_definition import FunctionDefinition
import json

def load_function_definitions(path: str) -> list[FunctionDefinition]:
    try:
        with open(Path(path), 'r', encoding=("utf-8")) as file:
            data = json.load(file)
            return [FunctionDefinition.model_validate(function) for function in data]

    except  FileNotFoundError:
        print(f"ERROR: file Not found error {path}")
        return []
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: invalid json file: {e}")
        return []