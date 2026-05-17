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
    
    full_prompt = f"""
        You are a function selector.
        Available functions:
        {functions}
        User prompt:
        {prompt}
        Task:
        Choose the single best function that matches the user's request.
        Rules:
        - Return only the function name
        - Do NOT explain your answer
        - Do NOT add extra text
        - The function name must exist in the available functions list
        Best function:
        """

    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []
    allowed_names: List[str] = [f['name']for f in functions]

    for _ in range(15):
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

    full_prompt = f"""
        Extract parameters for this function:
        {choosen['name']}
        Parameters needed:
        {json.dumps(parametres, indent=2)}
        User prompt:
        {prompt}
        Task:
        choose just the parametres and return a json string of those parametre
        Rules:
        Return ONLY this exact JSON structure with correct parameter names:
        {json.dumps({k: '?' for k in parametres.keys()})}
        json string: 
        """

    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    
    for _ in range(15):
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

