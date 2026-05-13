from nntplib import decode_header
import token
import torch
from llm_sdk.llm_sdk import Small_LLM_Model
from .models.function_call import FunctionCall

def main() -> None:
    model = Small_LLM_Model()
    # with open("function_calling_tests.json", 'r') as file:3
    # for text in tests:
        # token_ids = model.encode(text)
        # print(f"token_ids -------> {token_ids}")
        # for token in token_ids[0]:
            # decoded = model.decode([token.item()])
            # print(token.item(), "-->", repr(decoded))
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
    input_ids = model.encode(text2)
    input_ids = input_ids.squeeze().tolist()
    # input_ids_tensor = torch.tensor([input_ids])

    print("Promt:")
    print(text2)

    logits = model.get_logits_from_input_ids(input_ids)
    
    next_token_logits = logits[0]
    next_token_logits = torch.tensor(next_token_logits)
    predicted_token_id = torch.argmax(next_token_logits).item()
    print(f"prdicted-------->{type(predicted_token_id)}")
    print(f"next_token-------->{type(next_token_logits)}")
    predicted_token = model.decode([predicted_token_id])
    print(repr(predicted_token))
    # generated_ids = input_ids.copy()

    # for _ in range(10):
    #     logits = model.get_logits_from_input_ids(generated_ids)
    #     next_token_logits = logits[0][-1]
    #     next_token_id = torch.argmax(next_token_logits).item()
    #     generated_ids.append(next_token_id)
    # result = model.decode(generated_ids)
    # print(result)
    # print(f"type of logits ----> {type(logits)}")
    # predicted_token_id = torch.argmax(next_token_logits).item()
    # print(f"Logits shape: {logits}") 
    # predicted_text = model.decode([predicted_token_id])
    # print(repr(predicted_text))


main()