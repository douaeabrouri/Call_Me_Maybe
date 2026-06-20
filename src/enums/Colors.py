from enum import Enum


class Colors(Enum):
    """ANSI escape code colors for terminal
    output formatting.

    Each member represents an ANSI escape
    sequence that can be used
    to colorize text printed to the terminal.
    This enum provides a
    convenient way to apply and reset color formatting
    in terminal output.

    Attributes:
        RED (str): ANSI escape code for red text.
        GREEN (str): ANSI escape code for green text.
        YELLOW (str): ANSI escape code for yellow text.
        PURPLE (str): ANSI escape code for purple text.
        RESET (str): ANSI escape code to reset all formatting to default.

    Example:
        >>> print(f"{Colors.RED.value}Error!{Colors.RESET.value}")
        >>> print(f"{Colors.GREEN.value}Success!{Colors.RESET.value}")
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    PURPLE = "\033[35m"
    RESET = "\033[0m"
