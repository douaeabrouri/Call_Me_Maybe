from torch.cuda.memory import get_per_process_memory_fraction
from src.models.function_definition import FunctionDefinition
from src.utils.file_loader import load_function_definitions
from llm_sdk.llm_sdk import Small_LLM_Model
from torch.cuda import _get_generator
from typing import List
from src.utils.visualizer import GenerationVisualizer
import torch
import json
import sys

def get_numbers_from_prompt(prompt: str) -> set:
    import re as re_module

    return set(re_module.findall(r"\d+", prompt))


def is_garbage_prompt(prompt: str) -> bool:
    import re

    words = prompt.split()
    real_words = [w for w in words if re.match(r"^[a-zA-Z]{2,}$", w)]
    return len(real_words) < 2



def choose_function(
    prompt: str, model, functions: List[FunctionDefinition], vocab
) -> str:

    if is_garbage_prompt(prompt):
        return "NO_MATCH"

    allowed_names: List[str] = [f["name"] for f in functions]
    full_prompt = f"Functions: {allowed_names}\nRequest: '{prompt}'\nFunction name:"

    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []

    for _ in range(8):
        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        next_token_logits = torch.tensor(logits)
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids)
        for name in allowed_names:
            if name in decoded_text:
                return name
    return "NO_MATCH"

def is_valid_json_prefix(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    if not s.startswith("{"):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError as e:
        msg = str(e)
        if any(x in msg for x in ["Expecting", "Unterminated", "EOF", "end of data"]):
            return True
        return False

def extract_parameters(
    prompt: str,
    model,
    choosen: dict,
    valid_json_chars: set,
    id_to_token,
    visualize: bool = False,
) -> dict:
    parametres = choosen["parameters"]
    params_with_types = json.dumps(
        {k: v["type"] for k, v in parametres.items()}, indent=2
    )
    expected = json.dumps({k: v["type"] for k, v in parametres.items()})
    # expected_chars = len(expected)
    # max_tokens = max(8, expected_chars // 2)
    # print(f"max_tokens: {max_tokens}")

    if 'regex' not in parametres:
        full_prompt = f"""
        You are a parameter extractor.
        Function:
        {choosen['name']}
        Description:
        {choosen['description']}
        User request:
        {prompt}
        Rules:
        - Extract ONLY parameter values
        - No explanations, no extra keys
        Expected format:
        {expected}
        JSON:
        """
    else:
        full_prompt = f"""
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
        - No explanations, no extra keys
        - Extract values DIRECTLY from the user request
        - regex must describe WHAT should be replaced
        AND for the regex and replacement:
        Example 1:
        Replace all digits in "abc123" with X
        JSON:
            "source_string": "abc123",
            "regex": "[0-9]+",
            "replacement": "X"
        Example 2:
        Replace all vowels in "hello"
        JSON:
            "source_string": "hello",
            "regex": "[aeiouAEIOU]",
            "replacement": "*"
        Expected format:
        {expected}
        JSON:
        """

    viz = GenerationVisualizer(enabled=visualize)
    viz.reset(prompt)

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    current_json = ""
    blocked = 0
    allowed = 0
    len_para = len(parametres)
    max_tokens = 15 + (len_para * 10)
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        logits_tensor = torch.tensor(logits)
        needs_constraint = (
            current_json == ""
            or current_json.rstrip().endswith("{")
            or current_json.rstrip().endswith(",")
            or current_json.rstrip().endswith(":")
            or current_json.rstrip().endswith(": ")
        )
        if needs_constraint:
            for token_string, token_id in valid_json_chars:
                token = current_json + token_string
                if not is_valid_json_prefix(token):
                    logits_tensor[token_id] = float("-inf")
                    blocked += 1
                else:
                    allowed += 1
        next_token_id = int(torch.argmax(logits_tensor).item())
        generate_ids.append(next_token_id)
        token = id_to_token.get(next_token_id, "").replace("Ġ", " ").replace("Ċ", "\n")
        current_json += token
        viz.update(current_json, token, blocked, allowed)
        if current_json.strip().endswith("}"):
            break
        
    result = {}
    try:
        result = json.loads(current_json.strip())
        success: bool = True
    except json.JSONDecodeError:
        success = False
    viz.finish(current_json, success)
    return result


def validate_parameters(parameters: dict, function_definition: dict) -> bool:
    try:
        expected_params = function_definition["parameters"]
        for name, info in expected_params.items():
            if name not in parameters:
                return False
            expected_type = info["type"]
            if expected_type == "string":
                if not isinstance(parameters[name], str):
                    return False
            elif expected_type == "number":
                if not isinstance(parameters[name], (int, float)):
                    return False
            elif expected_type == "boolean":
                if not isinstance(parameters[name], bool):
                    return False
            else:
                raise ValueError(f"Unsupported parameter type: {expected_type}")
    except ValueError as e:
        print(f"Validation error: {e}")
        return False
    return True


def cast_parameters(params: dict, function_def: dict) -> dict:

    expected = function_def["parameters"]
    for name, info in expected.items():
        if name not in params:
            continue
        expected_type = info["type"]
        try:
            if expected_type == "string":
                params[name] = str(params[name])
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
