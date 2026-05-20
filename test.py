import re 

text = "hello/nworld {}"

match = re.search('h', text, re.DOTALL)
print(type(re.DOTALL))

if match:
    print("Done!")
else:
    print('Faild')