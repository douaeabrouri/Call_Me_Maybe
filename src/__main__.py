
from src.llm.generator import extract_parameters, choose_function, cast_parameters, validate_parameters
from src.utils.file_loader import load_function_definitions
from src.pipeline.function_caller import function_caller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder.token_filter import is_valid_regex, look_like_just_copy, is_valid_replacement
from pathlib import Path
from typing import List
import torch
from src.enums.Colors import Colors

def fix_regex(parameters: dict, prompt: str) -> dict:
    import re
    if 'regex' not in parameters:
        return parameters
    
    val = parameters['regex']
    
    if len(val) > 20 and 'source_string' in parameters and val == parameters['source_string']:
        match = re.search(r"(?:word|replace|substitute)\s+['\"]?(\w+)['\"]?\s+with", prompt, re.IGNORECASE)
        if match:
            parameters['regex'] = re.escape(match.group(1))
    
    return parameters

def fix_replacement(parameters: dict, prompt: str) -> dict:
    import re
    if "replacement" not in parameters:
        return parameters

    val = parameters["replacement"]
    WORD_TO_CHAR = {
        "asterisks": "*",
        "asterisk": "*",
        "stars": "*",
        "star": "*",
        "underscore": "_",
        "dash": "-",
        "empty": "",
        "nothing": "",
        "blank": "",
        "space": " ",
        "hash": "#",
        "pound": "#",
        "x": "X",
    }
    if val.lower() in WORD_TO_CHAR:
        parameters["replacement"] = WORD_TO_CHAR[val.lower()]
        return parameters
    match = re.search(r"with\s+'([^']+)'", prompt, re.IGNORECASE)
    if match:
        parameters["replacement"] = match.group(1)
        return parameters
    match = re.search(r"with\s+([A-Za-z0-9_*#@!]+)", prompt, re.IGNORECASE)
    if match:
        parameters["replacement"] = match.group(1)
    return parameters

def main() -> None:

    import json
    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    with open(path, 'r') as file:
        vocab = json.load(file)
    try:
        data, function  = load_function_definitions("functions_definition.json")
        if not data:
            print(f"{Colors.RED.value}ERROR: No function definitions found in 'functions_definition.json' {Colors.RESET.value}")
            return
    except FileNotFoundError:
        print(f"{Colors.RED.value}ERROR Functions_definition.json not found {Colors.RESET.value}")
        return
    except json.JSONDecodeError as e:
        print(f"{Colors.RED.value}ERROR functions_definition.json is malformed: {e} {Colors.RESET.value}")
        return 
    functions = {
        f["name"] : f["description"] for f in data
    }
    INPUTS_FOLDER = "data/input/"

    try:
        with open(Path(INPUTS_FOLDER + "function_calling_tests.json"), 'r', encoding=('utf-8')) as f:
            import json
            folder = json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED.value}ERROR: function_calling_tests.json not found in {INPUTS_FOLDER} {Colors.RESET.value}")
        return
    except json.JSONDecodeError as e:
        print(f"{Colors.RED.value}ERROR: function_calling_tests.json in malformed: {e} {Colors.RESET.value}")
        return
    except KeyError as e:
        print(f"{Colors.RED.value}ERROR: Missing exception in function_calling_tests.json: {e} {Colors.RESET.value}")
        return
    
    prompts: List[str] = [f['prompt'] for f in folder]
    results: List[dict] = []
    prompt = "Reverse the string 'hello'"
    # for i, prompt in enumerate(prompts):

        # print(f"{Colors.PURPLE.value}\n[{i+1}/{len(prompts)}] Processing {Colors.RESET.value}: '{prompt}'")
    
    try:
        func = choose_function(prompt, model, data, vocab)
        if func == "NO_MATCH":
            print(f"{Colors.YELLOW.value}WARNING: No function chosen for prompt '{prompt}'{Colors.RESET.value}")
            results.append({"prompt": prompt, "error": f"No matching function'"})
            # continue
        choosen = next((f for f in data if f['name'] == func), None)
        if choosen is None:
            print(f"{Colors.YELLOW.value}WARNING: Function '{func}' not found in definitions{Colors.RESET.value}")
            results.append({"prompt": prompt, "error": f"Function '{func}' not defined"})
            # continue
    except Exception as e:
        print(f"{Colors.RED.value}ERROR: Failed to choose function for prompt '{prompt}': {e}{Colors.RESET.value}")
        results.append({'prompt': prompt, 'error': f"Failed to chosen function: {e}"})
        # continue
    try:
        para = extract_parameters(prompt, model, choosen, vocab)
        if not para:
            print(f"{Colors.YELLOW.value}WARNING: No parameters extracted for prompt '{prompt}' {Colors.RESET.value}")
    except Exception as e:
        print(f"{Colors.RED.value}ERROR: Failed to extract parameters for prompt '{prompt}': {e} {Colors.RESET.value}")
        results.append({"prompt": prompt, "error": str(e)})
        # continue
    if 'regex' in para:
        para = fix_regex(para, prompt)
    if 'replacement' in para:
        para = fix_replacement(para, prompt)
    
    try:
        para = cast_parameters(para, choosen)
        if not validate_parameters(para, choosen):
            print(f"{Colors.YELLOW.value}WARNING: Extracted parameters are not valid for prompt '{prompt}' {Colors.RESET.value}")
            para = {}
    except Exception as e:
        print(f"{Colors.RED.value}ERROR: Failed to cast/validate parameters for prompt '{prompt}': {e} {Colors.RESET.value}")
    
    try:
        res = function_caller(prompt, func, para)
        results.append(res)
    except Exception as e:
        print(f"{Colors.RED.value}ERROR: Faild to call function for prompt '{prompt}': {e} {Colors.RESET.value}")
        results.append({"prompt": prompt, "error": str(e)})
    try:
        with open(Path("data/output/" + "function_calling_results.json"), "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"{Colors.RED.value}ERROR: Faild to write output: {e} {Colors.RESET.value}")

main()

#   {
#     "name": "fn_add_numbers",
#     "description": "Add two numbers together and return their sum.",
#     "parameters": {
#       "a": {
#         "type": "number"
#       },
#       "b": {
#         "type": "number"
#       }
#     },
#     "returns": {
#       "type": "number"
#     }
#   },