import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

print("Tokens 7900 - 8100 decoded (where NLL was 7.18 nats):")
print(repr(enc.decode(tokens[7900:8100].tolist())))

print("\n" + "="*80 + "\n")

print("Tokens 8400 - 8600 decoded (where NLL was 9.75 nats):")
print(repr(enc.decode(tokens[8400:8600].tolist())))
