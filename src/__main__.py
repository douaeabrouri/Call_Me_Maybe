from src.llm.tokenizer import choose_function
from typing import List
from src.utils.file_loader import load_function_definitions
from src.models.function_definition import FunctionDefinition


def main() -> None:
    data, function  = load_function_definitions("functions_definition.json")
    functions = {
        f["name"] : f["description"] for f in data
    }
    text = "What is the sum of 2 and 3?"
    full_prompt = f"Available functions: {function} and the prompt is {text} Return ONLY the best function name:"

    print(f"allowed_name: {functions}")
    print(choose_function(full_prompt, data))
main()