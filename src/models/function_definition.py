from pydantic import BaseModel


class ParameterType(BaseModel):
    """Represents the type declaration of a single parameter or return value.

    Attributes:
        type (str): The declared type as a string (e.g., "string",
            "number", "boolean").
    """
    type: str


class FunctionDefinition(BaseModel):
    """Defines the schema of a callable function for the LLM pipeline.

    Describes a function's identity, purpose, expected input parameters,
    and return type. Used by the pipeline to match user prompts to the
    correct function and validate extracted arguments.

    Attributes:
        name (str): The unique identifier of the function.
        description (str): A human-readable description of what the
            function does, used by the model to match user intent.
        parameters (dict[str, ParameterType]): A mapping of parameter
            names to their declared types.
        returns (ParameterType): The declared type of the function's
            return value.
    """
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType
