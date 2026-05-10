import json
from pathlib  import Path
from pydantic import BaseModel

class FunctionDefinition(BaseModel):
    BASE_DIR = Path(__file__).parent.parent.parent
    json_path = BASE_DIR / "data" / "input" / "functions_definition.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
    data_json: dict = {}
    if data['name'] == "fn_add_numbers":
        for f in data:
            if data != data['returnes']:
                
                
                          


def main() -> None:
    BASE_DIR = Path(__file__).parent.parent.parent
    json_path = BASE_DIR / "data" / "input" / "functions_definition.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
    for f in data:
        print(f"name: {f['name']}") 
        print(f"description: {f['description']}") 
        print(f"parametres: {f['parametres']}") 
        print(f"return: {f['returns']}")
        

if __name__ == "__main__":
    main()