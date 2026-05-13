from cmd import PROMPT
from typing import Any
from pydantic import BaseModel

# Prompt
#     ↓
# Choose function
#     ↓
# Generate parameters

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
