from typing import List
from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.file_loader import load_function_definitions
import torch
from src.models.function_definition import FunctionDefinition


""" khssni n buildi LLM prompt !!!"""
def choose_function(prompt: str, functions: List[FunctionDefinition]) -> str:

    model = Small_LLM_Model()
    
    input_ids = model.encode(prompt)
    generate_ids = []
    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = torch.argmax(next_token_logits).item()
        generate_ids.append(next_token_id)
        if model.decode(generate_ids) in functions:
            break
    return model.decode(generate_ids)
