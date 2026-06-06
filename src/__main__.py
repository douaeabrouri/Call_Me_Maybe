
from src.llm.generator import extract_parameters, choose_function, cast_parameters, validate_parameters
from src.utils.file_loader import load_function_definitions
from src.pipeline.function_caller import function_caller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder.token_filter import is_valid_regex, look_like_just_copy, is_valid_replacement
from pathlib import Path
from typing import List

def fix_regex(parameters: dict, prompt: str) -> dict:
    NATURAL_TO_REGEX = {
        "numbers": "\\d+",
        "digits": "\\d+", 
        "vowels": "[aeiouAEIOU]",
        "letters": "[a-zA-Z]+",
        "spaces": "\\s+",
    }
    if 'regex' not in parameters:
        return parameters
    val = parameters['regex']
    for word, pattern in NATURAL_TO_REGEX.items():
        if word in prompt.lower() and val not in NATURAL_TO_REGEX.values():
            parameters['regex'] = pattern
            break
    return parameters

def fix_replacement(parameters: dict, prompt: str) -> dict:
    import re
    if "replacement" not in parameters:
        return parameters

    match = re.search(r'with\s+([A-Za-z_*]+)', prompt, re.IGNORECASE)

    if match:
        parameters["replacement"] = match.group(1)

    return parameters

def main() -> None:
    import json
    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    with open(path, 'r') as file:
        vocab = json.load(file)
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
    # prompt: str = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
        func = choose_function(prompt, model, data, vocab)
        choosen = next((f for f in data if f['name'] == func), data[0])
        para = extract_parameters(prompt, model, choosen, vocab)
        if 'regex' in para:
            para = fix_regex(para, prompt)
        if 'replacement' in para:
            para = fix_replacement(para, prompt)
        if not validate_parameters(para, choosen):
            para = {}
        para = cast_parameters(para, choosen)
        res = function_caller(prompt, func, para)
        results.append(res)
    with open(Path("data/output/" + "function_calling_results.json"), "w") as f:
        json.dump(results, f, indent=4)
main()