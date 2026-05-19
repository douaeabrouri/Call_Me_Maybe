import json
from llm_sdk.llm_sdk import Small_LLM_Model
from src.llm.generator import choose_function, extract_parametres


def valid_token(s: str) -> None:
    model = Small_LLM_Model() 
    path = model.get_path_to_vocab_file()
    with open(path,'r') as file:
        vocab = json.load(file)
    
    