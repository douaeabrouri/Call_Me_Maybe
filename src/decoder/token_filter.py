import re as re_module


def is_valid_regex(pattern: str) -> bool:
    try:
        re_module.compile(pattern)
        return True
    except re_module.error:
        return False


def is_valid_replacement(pattern: str) -> bool:
    try:
        re_module.compile(pattern)
        return True
    except re_module.error:
        return False


def look_like_just_copy(regex: str, prompt: str) -> bool:
    return regex in prompt and not any(
        c in regex for c in [".", "*", "+", "?", "[", "]", "\\", "^", "$", "(", ")"]
    )
