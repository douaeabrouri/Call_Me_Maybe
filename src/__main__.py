
from src.decoder.token_filter import is_valid_regex, look_like_just_copy, is_valid_replacement
from src.llm.generator import extract_parameters, choose_function, cast_parameters, validate_parameters
from src.utils.file_loader import load_function_definitions
from src.pipeline.function_caller import function_caller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.enums.Colors import Colors
from pathlib import Path
from typing import List
import torch

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
            print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} No function definitions found in 'functions_definition.json'")
            return
        unknow_function = {"name": "Unknow_function", "description": "No matching function found for the given prompt", "parameters": {}, "return": {}}
        data.append(unknow_function)
    except FileNotFoundError:
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} Functions_definition.json not found")
        return
    except json.JSONDecodeError as e:
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} functions_definition.json is malformed: {e}")
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
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} function_calling_tests.json not found in {INPUTS_FOLDER}")
        return
    except json.JSONDecodeError as e:
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} function_calling_tests.json in malformed: {e}")
        return
    except KeyError as e:
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} Missing exception in function_calling_tests.json: {e}")
        return
    
    prompts: List[str] = [f['prompt'] for f in folder]
    results: List[dict] = []
    for i, prompt in enumerate(prompts):
        if prompt.strip() == "":
            print(f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} EMPTY PROMPT")
            results.append({"prompt": prompt, "error": "Empty prompt"})
            continue
        try:
            func = choose_function(prompt, model, data, vocab)
            if func == "NO_MATCH" or func is None:
                print(f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} No function chosen for prompt '{prompt}'")
                results.append({"prompt": prompt, "error": f"No matching function'"})
                continue
            choosen = next((f for f in data if f['name'] == func), None)
            if choosen is None:
                results.append({"prompt": prompt, "error": f"Function '{func}' not defined"})
                continue
        except Exception as e:
            print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} Failed to choose function for prompt '{prompt}': {e}")
            results.append({'prompt': prompt, 'error': f"Failed to choosen function: {e}"})
            continue
        try:
            para = extract_parameters(prompt, model, choosen, vocab, visualize = True)
            if not para:
                print(f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} No parameters extracted for prompt '{prompt}'")
        except Exception as e:
            print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} Failed to extract parameters for prompt '{prompt}': {e}")
            results.append({"prompt": prompt, "error": str(e)})
            continue
        if 'regex' in para:
            para = fix_regex(para, prompt)
        if 'replacement' in para:
            para = fix_replacement(para, prompt)
        para = cast_parameters(para, choosen)
        if not validate_parameters(para, choosen):
            print(f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} Extracted parameters are not valid for prompt '{prompt}'")
            para = {}        
        res = function_caller(prompt, func, para)
        results.append(res)
    try:
        with open(Path("data/output/" + "function_calling_results.json"), "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"{Colors.RED.value}ERROR:{Colors.RESET.value} Faild to write output: {e}")

main()
