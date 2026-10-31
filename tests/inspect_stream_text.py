import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

print("Tokens 0 - 200 decoded:")
print(repr(enc.decode(tokens[0:200].tolist())))

print("\nTokens 3000 - 3200 decoded:")
print(repr(enc.decode(tokens[3000:3200].tolist())))

print("\nTokens 4500 - 4700 decoded:")
print(repr(enc.decode(tokens[4500:4700].tolist())))
