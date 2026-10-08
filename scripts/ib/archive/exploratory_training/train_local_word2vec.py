"""Fast GPU Skip-Gram Negative Sampling (SGNS) Pretraining for GPT-2 Vocabulary.

Trains a 128-dimensional dense semantic embedding matrix directly from the local
OpenWebText token stream (data/ib_owt_gpt2/train.npy) in ~30 seconds on GPU.

Outputs:
  data/pretrained_embeddings_owt_128d.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/train.npy"))
    parser.add_argument("--output", type=Path, default=Path("data/pretrained_embeddings_owt_128d.pt"))
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1500000,
                        help="Number of tokens from corpus to use for training")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--k-neg", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.max_tokens:,} tokens from {args.data}...")
    tokens_np = np.load(args.data, mmap_mode="r")[:args.max_tokens]
    tok_tensor = torch.from_numpy(np.array(tokens_np, dtype=np.int64)).to(device)

    print(f"Generating Skip-Gram context pairs (window [-2, -1, +1, +2])...")
    t0 = time.perf_counter()
    pairs = []
    for w in [-2, -1, 1, 2]:
        if w > 0:
            c = tok_tensor[:-w]
            ctx = tok_tensor[w:]
        else:
            c = tok_tensor[-w:]
            ctx = tok_tensor[:w]
        pairs.append((c, ctx))

    all_c = torch.cat([p[0] for p in pairs])
    all_ctx = torch.cat([p[1] for p in pairs])
    N_pairs = all_c.shape[0]
    print(f"Generated {N_pairs:,} training pairs in {time.perf_counter() - t0:.2f}s")

    # Word2Vec Embeddings
    embed_in = nn.Embedding(args.vocab_size, args.d_model).to(device)
    embed_out = nn.Embedding(args.vocab_size, args.d_model).to(device)
    nn.init.normal_(embed_in.weight, std=0.05)
    nn.init.normal_(embed_out.weight, std=0.05)

    optimizer = torch.optim.Adam(
        list(embed_in.parameters()) + list(embed_out.parameters()),
        lr=args.lr
    )

    print(f"Training 128-dim SGNS across {args.epochs} epochs on {device}...")
    t_train = time.perf_counter()

    for epoch in range(args.epochs):
        perm = torch.randperm(N_pairs, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N_pairs, args.batch_size):
            idx = perm[i:i + args.batch_size]
            c = all_c[idx]
            ctx = all_ctx[idx]
            neg = torch.randint(0, args.vocab_size, (c.shape[0], args.k_neg), device=device)

            v_c = embed_in(c)               # [B, d]
            v_ctx = embed_out(ctx)          # [B, d]
            v_neg = embed_out(neg)          # [B, K, d]

            pos_score = (v_c * v_ctx).sum(dim=-1)                        # [B]
            neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)  # [B, K]

            loss_pos = -torch.log(torch.sigmoid(pos_score).clamp_min(1e-7))
            loss_neg = -torch.log(torch.sigmoid(-neg_score).clamp_min(1e-7)).sum(dim=-1)
            loss = (loss_pos + loss_neg).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Elapsed: {time.perf_counter() - t_train:.1f}s")

    # Normalize embeddings to unit sphere
    normed_in = F.normalize(embed_in.weight.data, p=2, dim=-1)
    normed_out = F.normalize(embed_out.weight.data, p=2, dim=-1)
    # Combine in and out embeddings as standard in Word2Vec
    final_embeddings = 0.5 * (normed_in + normed_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embedding": final_embeddings.cpu(),
        "embed_in": normed_in.cpu(),
        "embed_out": normed_out.cpu(),
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
    }, args.output)

    print(f"\nSuccessfully saved pretrained embeddings to {args.output} "
          f"({args.output.stat().st_size / (1024**2):.2f} MB)")

    # Verify cosine similarity of key tokens
    print("\n--- Semantic Similarity Sanity Check ---")
    sim_the_a = F.cosine_similarity(final_embeddings[262:263], final_embeddings[257:258]).item()
    sim_the_an = F.cosine_similarity(final_embeddings[262:263], final_embeddings[281:282]).item()
    print(f"Cosine Similarity (' the' [262] vs ' a' [257]):   {sim_the_a:+.4f}")
    print(f"Cosine Similarity (' the' [262] vs ' an' [281]):  {sim_the_an:+.4f}")


if __name__ == "__main__":
    main()
