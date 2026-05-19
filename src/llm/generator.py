from ast import Dict
from typing import List
from torch.cuda import _get_generator
from torch.cuda.memory import get_per_process_memory_fraction
from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.file_loader import load_function_definitions
import torch
from src.models.function_definition import FunctionDefinition
import re
import json


def is_valid_json_prefix(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    try:
        json.loads(s)
        return True 
    except json.JSONDecodeError as e:
        return "Expecting" in str(e) or "Unterminated" in str(e)

def choose_function(prompt: str, model, functions: List[FunctionDefinition]) -> str:
    
    allowed_names: List[str] = [f['name']for f in functions]
    full_prompt = f"Functions: {allowed_names}\nRequest: '{prompt}'\nFunction name:"

    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []

    path = model.get_path_to_vocab_file()
    with open(path, 'r') as file:
        vocab = json.load(file)
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

def extract_parametres(prompt: str, model, choosen: dict, vocab) -> dict:
    
    parametres = choosen['parameters']
    params_with_types = json.dumps(
        {k: v['type'] for k, v in parametres.items()}, 
        indent=2
    )
    expected = json.dumps({k: v['type'] for k, v in parametres.items()})
    full_prompt = f"""
    You are a parameter extractor.
    
    Function:
    {choosen['name']}
    
    Expected parameters:
    {params_with_types}
    
    User request:
    {prompt}
    
    Rules:
    - Extract ONLY parameter values
    - DO NOT execute the function
    - Return ONLY valid JSON
    - Use exact parameter names
    - No explanations
    - No extra keys
    
    Expected format:
    {expected}
    
    JSON:
    """

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []

    sample = list(vocab.items())[0]
    current_json = ""
    id_to_token = {
        v: k for k, v in vocab.items()
    }
    for _ in range(20):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        logits_tensor = torch.tensor(logits)
        for token_string, token_id in vocab.items():
            test = current_json + token_string
            if not is_valid_json_prefix(test):
                logits_tensor[token_id] = float('-inf')
        # next_token_logits = torch.tensor(logits)
        next_token_id = int(torch.argmax(logits_tensor).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids)
        token = id_to_token.get(next_token_id, '')
        # print(f"Generated token: '{token}'")
        current_json += token
        # print(model.decode(generate_ids))
        if current_json.strip().endswith('}'):
            break
        match = re.search(r'\{.*?\}', decoded_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
    return {}
