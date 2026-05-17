from ast import Dict
from typing import List
from torch.cuda.memory import get_per_process_memory_fraction
from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.file_loader import load_function_definitions
import torch
from src.models.function_definition import FunctionDefinition
import re
import json


def choose_function(prompt: str, model, functions: List[FunctionDefinition]) -> str:
    
    allowed_names: List[str] = [f['name']for f in functions]
    full_prompt = f"Functions: {allowed_names}\nRequest: '{prompt}'\nFunction name:"

    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []

    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids)
        for name in allowed_names:
            if name in decoded_text:
                return name
    return model.decode(generate_ids)


def extract_parametres(prompt: str, model, choosen: dict) -> dict:
    
    parametres = choosen['parameters']
    params_with_types = json.dumps(
    {k: v['type'] for k, v in parametres.items()}, 
    indent=2
)
    expected = json.dumps({k: '?' for k in parametres.keys()})
    full_prompt = f"""Function: {choosen['name']}
    Request: '{prompt}'
    Extract the parameter values from the request and return ONLY this JSON with real values (not ?):
    {expected}
    JSON:"""

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    
    for _ in range(20):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids) 
        match = re.search(r'\{.*?\}', decoded_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
    return {}

