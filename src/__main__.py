from llm_sdk.llm_sdk import Small_LLM_Model

def main() -> None:
    model = Small_LLM_Model()
    # print("modeal loaded successufully")
    # text: str = "c est quoi la somme de 3 et 5?"
    # token_ids  = model.encode(text)
    # print(f"original text -----> {text}")
    # print(f"token ids -----> {token_ids}")
    # decode_txt = model.decode(token_ids)
    # print(f"decode ------->{decode_txt}")

    # with open("function_calling_tests.json", 'r') as file:
    tests = [
        "{",
        "}",
        '"',
        '"name"',
        ":",
        ",",
        "fn_add_numbers",
        "hello",
        "42",
    ]
    for text in tests:
        token_ids = model.encode(text)
        print(f"Text ------------>{text}")
        print(f"Token ids ----------->{token_ids}")
        decoded = model.decode(token_ids)
        print(f"Decode ------------->{decoded}")
        print("-" * 40)


if __name__ == "__main__":
   main()