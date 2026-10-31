import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

print("Tokens 16900 - 17200 decoded:")
print(repr(enc.decode(tokens[16900:17200].tolist())))

print("\n" + "="*80 + "\n")

print("Tokens 17900 - 18200 decoded:")
print(repr(enc.decode(tokens[17900:18200].tolist())))
