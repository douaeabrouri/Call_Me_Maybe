from typing import Any
from pydantic import BaseModel


class FunctionCall(BaseModel):
    """Represents a resolved function call with its extracted arguments.

    Stores the original user prompt alongside the matched function name
    and its extracted parameter values, ready for execution.

    Attributes:
        prompt (str): The original user input that triggered the function call.
        name (str): The name of the matched function to invoke.
        parameters (dict[str, Any]): The extracted parameter values mapped
            by parameter name.
    """
    prompt: str
    name: str
    parameters: dict[str, Any]
