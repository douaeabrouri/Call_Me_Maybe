from uu import decode
from src.llm.generator import extract_parametres, choose_function
from src.models.function_definition import FunctionDefinition
import json
import torch
from typing import List

def function_caller(user_prompt: str, function_name: str, function_parametre: dict) -> dict:   
    return {
        'prompt': user_prompt,
        'name': function_name,
        'parametres': function_parametre
    }