from src.llm.tokenizer import choose_function
from typing import List
from src.utils.file_loader import load_function_definitions
from src.models.function_definition import FunctionDefinition
from pathlib import Path
from src.llm.generator import  choose_function, extract_parametres
from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    data, function  = load_function_definitions("functions_definition.json")
    functions = {
        f["name"] : f["description"] for f in data
    }
    
    INPUTS_FOLDER = "data/input/"

    with open(Path(INPUTS_FOLDER + "function_calling_tests.json"), 'r', encoding=('utf-8')) as f:
        import json
        folder = json.load(f)
    
    p: List[str] = [f['prompt'] for f in folder]

    i = "What is the sum of 265 and 345?" 
    full_prompt = f"""
        You are a function selector.
        Available functions:
        {functions}
        User prompt:
        {i}
        Task:
        Choose the single best function that matches the user's request.
        Rules:
        - Return only the function name
        - Do NOT explain your answer
        - Do NOT add extra text
        - The function name must exist in the available functions list
        Best function:
        """
    model = Small_LLM_Model()

    for f in data:
        if choose_function(full_prompt, model, data) == f['name']:
            parametres = f['parameters']
            print(parametres) 
    secand_prompt = f"""
        Extract parameters for this function:
        {choose_function(full_prompt, model, data)}
        Parameters needed:
        {parametres}
        User prompt:
        {i}
        Task:
        choose just the parametres and return a json string of those parametre
        Rules:
        - Return ONLY a JSON object like: {{"a": ?, "b": ?}}
        - Return only the function name
        - Do NOT explain your answer
        - Do NOT add extra text
        json string: 
        """

    func = choose_function(full_prompt,model,data)
    para = extract_parametres(secand_prompt, model, data)
    print(para)

main()