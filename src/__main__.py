from src.llm.tokenizer import choose_function
from typing import List
from src.utils.file_loader import load_function_definitions
from src.models.function_definition import FunctionDefinition
from pathlib import Path
from src.llm.generator import  choose_function, extract_parametres
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

    prompts: List[str] = [f['prompt'] for f in folder]
    results: List[dict] = []
    for prompt in prompts:
        func = choose_function(prompt,model,data)
        para = extract_parametres(prompt, model, data)
        res = function_caller(prompt, func, para)
        results.append(res)
    print(results)

main()