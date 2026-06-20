def function_caller(
    user_prompt: str, function_name: str, function_parametre: dict
) -> dict:
    """Assembles a function call dictionary from its core components.

    Packages the user prompt, matched function name, and extracted
    parameters into a single dictionary representing a resolved
    function call.

    Args:
        user_prompt (str): The original user input that triggered
            the function call.
        function_name (str): The name of the matched function to invoke.
        function_parametre (dict): The extracted parameter values
            mapped by parameter name.

    Returns:
        dict: A dictionary with keys "prompt", "name", and "parameters"
            representing the fully resolved function call.
    """
    return {
        "prompt": user_prompt,
        "name": function_name,
        "parameters": function_parametre,
    }
