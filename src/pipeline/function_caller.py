def function_caller(
    user_prompt: str, function_name: str, function_parametre: dict
) -> dict:
    return {
        "prompt": user_prompt,
        "name": function_name,
        "parameters": function_parametre,
    }
