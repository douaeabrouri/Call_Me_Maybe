from inspect import Parameter
import json
from pathlib  import Path
from pydantic import BaseModel, Field

class ParameterType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parametres: dict[str, ParameterType]
    returns: ParameterType


# def main() -> None:
#     BASE_DIR = Path(__file__).parent.parent.parent
#     json_path = BASE_DIR / "data" / "input" / "functions_definition.json"
#     with open(json_path, 'r') as file:
#         data = json.load(file)
#     for f in data:
#         print(f"name: {f['name']}") 
#         print(f"description: {f['description']}") 
#         print(f"parametres: {f['parametres']}") 
#         print(f"return: {f['returns']}")
        

# if __name__ == "__main__":
#     main()