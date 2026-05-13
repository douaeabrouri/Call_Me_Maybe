
from pydantic import BaseModel

class ParameterType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parametres: dict[str, ParameterType]
    returns: ParameterType
