from typing import List
from torch.cuda import _get_generator
from torch.cuda.memory import get_per_process_memory_fraction
from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.file_loader import load_function_definitions
import torch
from src.models.function_definition import FunctionDefinition
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
    - choose exactly regex and replacement
    Expected format:
    {expected}
    JSON:
    """
    # For example:
    # Function:
    # fn_substitute_string_with_regex
    # Description:
    # Replace all occurrences matching a regex pattern in a string.
    # Parameters:
    # source_string: original text
    # regex: regex pattern to match
    # replacement: text that replaces each match

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    current_json = ""
    id_to_token: dict = {v: k for k, v in vocab.items()}
    for _ in range(150):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        """Test for the regex and replacement tokens"""
        logits_tensor = torch.tensor(logits)
        topk = torch.topk(logits_tensor, 5)
        for score, token_id in zip(topk.values.tolist(),
                           topk.indices.tolist()):
                token = id_to_token.get(token_id, "")
                candidate = current_json + token.replace('Ġ', ' ').replace('Ċ', '\n')
                if not is_valid_json_prefix(candidate):
                   logits_tensor[token_id] = float('-inf')
        next_token_id = int(torch.argmax(logits_tensor).item())
        generate_ids.append(next_token_id)
        token = id_to_token.get(next_token_id, '').replace('Ġ', ' ').replace('Ċ', '\n')
        current_json += token
        if current_json.strip().endswith('}'):
            break
    print(repr(current_json))
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
        if expected_type == "number":
            try:
                params[name] = int(params[name])
            except (ValueError, TypeError) as e:
                print(f"ERROR: {e}")
                sys.exit(0)
        elif expected_type == "floating":
            try:
                params[name] = float(params[name])
            except (ValueError, TypeError) as e:
                print(f"ERROR: {e}")
                sys.exit(0)
        elif expected_type == "boolean":
            if isinstance(params[name], str):
                params[name] = (params[name].lower() == "true")
    return params