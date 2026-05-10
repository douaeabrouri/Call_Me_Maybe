from nntplib import decode_header
import token
import torch
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
        # print(f"token_ids -------> {token_ids}")
        for token in token_ids[0]:
            decoded = model.decode([token.item()])
            print(token.item(), "-->", repr(decoded))
    # path = model.get_path_to_vocab_file()
    # test = model.encode('"parametres"')
    # test2 = model.encode('"name"')
    # print(path)
    # print(test)
    # print(type(test2))
    
    # text = "fn_add_numbers"
    # token_ids = model.encode(text)
    # print(f"token_ids------->{token_ids}")
    # for token in token_ids[0]:
    #     decoded = model.decode([token.item()])
    #     print(token.item(), "-->", repr(decoded))
    text2: str = "What is the sum of"
    input_ids =model.encode(text2)
    input_ids = input_ids.squeeze().tolist()
    print(f"Input ids: {input_ids}")
    logits = model.get_logits_from_input_ids(input_ids)
    print(f"type of logits ----> {type(logits)}")
    next_token_logits = logits
    # predicted_token_id = torch.argmax(next_token_logits).item()
    print(f"testttttt logit: -------> {next_token_logits}")
    # print(f"Logits shape: {logits}")
    # predicted_text = model.decode([predicted_token_id])
    # print(repr(predicted_text))




if __name__ == "__main__":
   main()