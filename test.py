import json 

if __name__ == "__main__":
    with open("test.json", 'r') as file:
        data = json.load(file)
    print(data['school'])
    string: str = data['school']
    encoded_bytes = string.encode('utf-8')
    decoded_str = encoded_bytes.decode('utf-8')
    print(encoded_bytes)
    print(decoded_str)
