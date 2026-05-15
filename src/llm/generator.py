from typing import List
from torch.cuda.memory import get_per_process_memory_fraction
from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.file_loader import load_function_definitions
import torch
from src.models.function_definition import FunctionDefinition

# model = Small_LLM_Model()
def choose_function(full_prompt: str, model,functions: List[FunctionDefinition]) -> str:


    input_ids: List[int] = model.encode(full_prompt)[0].tolist()
    generate_ids: list[int] = []
    allowed_names: List[str] = [f['name'] for f in functions]

    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids)
        for name in allowed_names:
            if name in decoded_text:
                return name
    return model.decode(generate_ids)



def extract_parametres(full_prompt: str, model, functions: List[FunctionDefinition]) -> List[str]:
    
    input_id = model.encode(full_prompt)[0].tolist()
    generate_ids: List[int] = []
    # allowed_names: List[str] = [f['name'] for f in functions]
    
    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_id + generate_ids)
        next_token_logits = torch.tensor(logits)    
        next_token_id = int(torch.argmax(next_token_logits).item())
        generate_ids.append(next_token_id)
        decoded_text = model.decode(generate_ids) 
        if '}' in decoded_text:
            break 
    
    return model.decode(generate_ids)