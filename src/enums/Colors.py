from enum import Enum

class Colors(Enum) :
    RED =  "\033[31m"
    GREEN = "\033[32m"
    YELLOW  = "\033[33m"
    PURPLE = "\033[35m"
    RESET = "\033[0m"