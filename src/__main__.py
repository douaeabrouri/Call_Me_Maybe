
from typing import List
from src.utils.file_loader import load_function_definitions
from pathlib import Path
from src.llm.generator import extract_parameters, choose_function, cast_parameters, validate_parameters
from src.pipeline.function_caller import function_caller
from llm_sdk.llm_sdk import Small_LLM_Model



def main() -> None:
    model = Small_LLM_Model()
    INPUTS_FOLDER = "data/input/"
    data, function  = load_function_definitions("functions_definition.json")
    functions = {
        f["name"] : f["description"] for f in data
    }

    with open(Path(INPUTS_FOLDER + "function_calling_tests.json"), 'r', encoding=('utf-8')) as f:
        import json
        folder = json.load(f)
    path = model.get_path_to_vocab_file()
    with open(path, 'r') as file:
        vocab = json.load(file)

    prompts: List[str] = [f['prompt'] for f in folder]
    results: List[dict] = []
    for prompt in prompts:
        func = choose_function(prompt, model, data)
        choosen = next((f for f in data if f['name'] == func), data[0])
        para = extract_parameters(prompt, model, choosen, vocab)
        para = cast_parameters(para, choosen)
        if not validate_parameters(para, choosen):
            para = {}
        res = function_caller(prompt, func, para)
        results.append(res)
    with open(Path("data/output/" + "function_calling_results.json"), "w") as f:
        json.dump(results, f, indent=4)
main()
