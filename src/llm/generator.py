from torch.cuda.memory import get_per_process_memory_fraction
from src.models.function_definition import FunctionDefinition
from src.utils.file_loader import load_function_definitions
from llm_sdk.llm_sdk import Small_LLM_Model
from torch.cuda import _get_generator
from typing import List
import torch
import json
import sys


def is_valid_json_prefix(s: str) -> bool:  
    s = s.strip()
    if not s:
        return True
    if not s.startswith('{'):
        return False
    try:
        json.loads(s)
        return True 
    except json.JSONDecodeError as e:
        msg = str(e)
        if any(x in msg for x in [
            "Expecting",
            "Unterminated",
            "EOF",
            "end of data"
        ]):
            return True
        return False

def get_numbers_from_prompt(prompt: str) -> set:
    """Extract all number strings from prompt."""
    import re as re_module
    return set(re_module.findall(r'\d+', prompt))

def choose_function(prompt: str, model, functions: List[FunctionDefinition], vocab) -> str:
    
    allowed_names: List[str] = [f['name'] for f in functions]
    full_prompt = f"Functions: {allowed_names}\nRequest: '{prompt}'\nFunction name:"

    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []
    current_text = ""
    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids)
        current_text  += vocab.get(str(next_token_id), '')
        for name in allowed_names:
            if name in decoded_text:
                return name
    return current_text.strip()

def extract_parameters(prompt: str, model, choosen: dict, vocab) -> dict:
    parametres = choosen['parameters']
    params_with_types = json.dumps(
        {k: v['type'] for k, v in parametres.items()},
        indent = 2
    )
    expected = json.dumps({k: v['type'] for k, v in parametres.items()})

    full_prompt = f"""
    You are a parameter extractor.
    Function:
    {choosen['name']}
    Description:
    {choosen['description']}
    Expected parameters:
    {params_with_types}
    User request:
    {prompt}
    Rules:
    - Extract ONLY parameter values
    - DO NOT execute the function
    - Return ONLY JSON
    - No explanations, no extra keys
    - Extract values DIRECTLY from the user request
    - Use exact parameter names
    - If a parameter type is number, return a JSON number
    - regex must describe WHAT should be replaced
    - Return the values that would be passed to the function call.
    and for the regex and replacement:
    Example 1:
    Request:
    Replace all digits in "abc123" with X
    JSON:
        "source_string": "abc123",
        "regex": "[0-9]+",
        "replacement": "X"
    Example 2:
    Request:
    Replace all vowels in "hello" with "asterisk"
    JSON:
        "source_string": "hello",
        "regex": "[aeiouAEIOU]",
        "replacement": "*"
    Expected format:
    {expected}
    JSON:
    """

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    current_json = ""
    id_to_token: dict = {v: k for k, v in vocab.items()}
    valid_json_chars = set()
    for token_string, token_id in vocab.items():
        clean = token_string.replace('Ġ', ' ').replace('Ċ', '\n')
        if any(c in clean for c in '[]"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-_+ \\'):
           valid_json_chars.add((clean, int(token_id)))
    for _ in range(150):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        logits_tensor = torch.tensor(logits)
        for token_string, token_id in valid_json_chars:
            token = current_json + token_string
            if not is_valid_json_prefix(token):
                logits_tensor[token_id] = float('-inf')
        # next_token_id = int(torch.argmax(logits_tensor).item())
        probs = torch.softmax(logits_tensor, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        generate_ids.append(next_token_id)
        token = id_to_token.get(next_token_id, '').replace('Ġ', ' ').replace('Ċ', '\n')
        current_json += token
        if current_json.strip().endswith('}'):
            break
    try:
        return json.loads(current_json.strip())
    except json.JSONDecodeError:
        return {}

def validate_parameters(parameters: dict, function_definition: dict) -> bool:

    expected_params = function_definition['parameters']
    for name, info in expected_params.items():
        if name not in parameters:
            return False
        expected_type = info['type']
        if expected_type == "string":
           if not isinstance(parameters[name], str):
              return False
        elif expected_type == "number":
            if not isinstance(parameters[name], (int, float)):
               return False
        elif expected_type == "integer":
            if not isinstance(parameters[name], int):
                return False
        elif expected_type == "boolean":
            if not isinstance(parameters[name], bool):
               return False
    return True

def cast_parameters(params: dict, function_def: dict) -> dict:

    expected = function_def["parameters"]
    for name, info in expected.items():
        if name not in params:
            continue
        expected_type = info["type"]
        try:
            if expected_type == "integer":
                params[name] = int(str(params[name]))
            elif expected_type == "number":
                params[name] = int(float(params[name]))
            elif expected_type == "boolean":
                if isinstance(params[name], str):
                    params[name] = params[name].lower() == "true"
            else:
                params[name] = str(params[name])
        except (ValueError, TypeError) as e:
            print(f"Warning: could not cast {name}: {e}")
    return params