from src.llm.generator import (
    extract_parameters,
    choose_function,
    cast_parameters,
    validate_parameters,
)
from src.utils.file_loader import load_function_definitions
from src.pipeline.function_caller import function_caller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.enums.Colors import Colors
from pathlib import Path
from typing import List
import json


def fix_regex(parameters: dict, prompt: str) -> dict:
    import re

    if "regex" not in parameters:
        return parameters

    val = parameters["regex"]

    if (
        len(val) > 20
        and "source_string" in parameters
        and val == parameters["source_string"]
    ):
        match = re.search(
            r"(?:word|replace|substitute)\s+['\"]?(\w+)['\"]?\s+with",
            prompt,
            re.IGNORECASE,
        )
        if match:
            parameters["regex"] = re.escape(match.group(1))

    return parameters


def fix_replacement(parameters: dict, prompt: str) -> dict:
    import re

    if "replacement" not in parameters:
        return parameters

    val = parameters["replacement"]
    WORD_TO_CHAR = {
        "asterisks": "*",
        "asterisk": "*",
        "stars": "*",
        "star": "*",
        "underscore": "_",
        "dash": "-",
        "empty": "",
        "nothing": "",
        "blank": "",
        "space": " ",
        "hash": "#",
        "pound": "#",
        "x": "X",
    }
    if val.lower() in WORD_TO_CHAR:
        parameters["replacement"] = WORD_TO_CHAR[val.lower()]
        return parameters
    match = re.search(r"with\s+'([^']+)'", prompt, re.IGNORECASE)
    if match:
        parameters["replacement"] = match.group(1)
        return parameters
    match = re.search(r"with\s+([A-Za-z0-9_*#@!]+)", prompt, re.IGNORECASE)
    if match:
        parameters["replacement"] = match.group(1)
    return parameters


def main() -> None:
    model = Small_LLM_Model()
    try:
        data, _ = load_function_definitions("functions_definition.json")
        if not data:
            print(
                f"{Colors.RED.value}ERROR:{Colors.RESET.value} No "
                f"function definitions found in 'functions_definition.json'"
            )
            return
    except FileNotFoundError:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value}"
            f" Functions_definition.json not found"
        )
        return
    except json.JSONDecodeError as e:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
            f"functions_definition.json is malformed: {e}"
        )
        return
    INPUTS_FOLDER = "data/input/"
    try:
        with open(
            Path(INPUTS_FOLDER + "function_calling_tests.json"),
            "r",
            encoding=("utf-8")
        ) as f:
            folder = json.load(f)
    except FileNotFoundError:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
            f"function_calling_tests.json not found in {INPUTS_FOLDER}"
        )
        return
    except json.JSONDecodeError as e:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
            f"function_calling_tests.json in malformed: {e}"
        )
        return
    except KeyError as e:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
            f"Missing exception in function_calling_tests.json: {e}"
        )
        return

    path = model.get_path_to_vocab_file()
    with open(path, "r") as file:
        vocab = json.load(file)

    JSON_EXACT = {
        "{",
        "}",
        "[",
        "]",
        '"',
        ":",
        ",",
        " ",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ".",
        "-",
        "+",
        "e",
        "E",
        "n",
        "u",
        "l",
        "t",
        "r",
        "a",
        "f",
        "s",
    }
    valid_json_chars: set = set()

    for token_string, token_id in vocab.items():
        clean = token_string.replace("Ġ", " ").replace("Ċ", "\n")
        stripped = clean.strip()
        if stripped in JSON_EXACT:
            valid_json_chars.add((clean, int(token_id)))
    for token_string, token_id in vocab.items():
        clean = token_string.replace("Ġ", " ").replace("Ċ", "\n")
        stripped = clean.strip()
        if (
            stripped
            and len(stripped) <= 2
            and stripped.isascii()
            and stripped.isalnum()
        ):
            valid_json_chars.add((clean, int(token_id)))

    prompts: List[str] = [f["prompt"] for f in folder]
    results: List[dict] = []
    id_to_token: dict = {int(v): k for k, v in vocab.items()}
    ALL_JSON_TOKEN_IDS = {token_id for _, token_id in valid_json_chars}
    for i, prompt in enumerate(prompts):
        if not prompt.strip():
            print(f"{Colors.YELLOW.value}WARNING:"
                  f"{Colors.RESET.value} EMPTY PROMPT")
            results.append({"prompt": prompt, "error": "Empty prompt"})
            continue
        try:
            func = choose_function(prompt, model, data)
            if func == "NO_MATCH":
                print(
                    f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} "
                    f"No function chosen for prompt '{prompt}'")
                results.append({"prompt": prompt,
                                "error": "No matching function'"})
                continue

            choosen = next((f for f in data if f["name"] == func), None)
            if choosen is None:
                results.append(
                    {"prompt": prompt,
                     "error": f"Function '{func}' not defined"}
                )
                continue
        except Exception as e:
            print(e)
            print(
                f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
                f"Failed to choose function for prompt '{prompt}': {e}"
            )
            results.append(
                {"prompt": prompt, "error": f"Failed to choosen function: {e}"}
            )
            continue

        try:
            para = extract_parameters(
                prompt,
                model,
                choosen,
                valid_json_chars,
                id_to_token,
                ALL_JSON_TOKEN_IDS,
                visualize=True,
            )

            if para is None:
                para = {}

        except Exception as e:
            print(
                f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
                f"Failed to extract parameters for prompt '{prompt}': {e}"
            )
            results.append({"prompt": prompt, "error": str(e)})
            continue
        if "regex" in para:
            para = fix_regex(para, prompt)
        if "replacement" in para:
            para = fix_replacement(para, prompt)

        para = cast_parameters(para, choosen)
        if not validate_parameters(para, choosen):
            print(
                f"{Colors.YELLOW.value}WARNING:{Colors.RESET.value} "
                f"Extracted parameters are not valid for prompt '{prompt}'"
            )
            para = {}
        res = function_caller(prompt, func, para)
        results.append(res)
    try:
        with open(
            Path("data/output/" + "function_calling_results.json"),
            "w"
        ) as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(
            f"{Colors.RED.value}ERROR:{Colors.RESET.value} "
            f"Faild to write output: {e}"
        )


main()
