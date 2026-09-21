"""Diagnostic Script: Per-Puzzle Fine-Grained Pondering Trajectory Analysis.
Answers the User's Critical Question:
"Why does the global average plateau at K=2?
Are there specific hard puzzles where deep pondering (K=16, 64, 256, 1024) produces massive gains,
which are being masked by simple puzzles that resolve in 2 steps?"

Analysis breakdown across all 1,000 test puzzles:
1. Puzzles where deep K (K=64/256/1024) beats K=2 (Deep Thinking Winners)
2. Puzzles where K=2 beats deep K (Overshoot / Degradation Cases)
3. Puzzles invariant to K (Already resolved by K=2)
4. Correlation between puzzle difficulty (clue count) and optimal pondering depth K_opt.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.train_ultra_deep_k1024_3000 import CheckpointedUltraDeepCBIMSudokuModel


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load our Best Adaptive Frontier Model
    ckpt_path = Path("results/adaptive_expanding_frontier_3000/Best_Adaptive_Frontier_d256.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("results/adaptive_expanding_frontier_3000/Final_Adaptive_Frontier_d256.pt")

    print(f"Loading model from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CheckpointedUltraDeepCBIMSudokuModel(vocab_size=11, d_channels=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")
    num_test = len(test_in)

    horizons = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    clue_counts = (test_in > 1).sum(axis=1)

    print(f"Scanning all {num_test} puzzles across {len(horizons)} horizons...", flush=True)

    # Store per-puzzle accuracy: [num_test, len(horizons)]
    puzzle_accs = np.zeros((num_test, len(horizons)))
    puzzle_losses = np.zeros((num_test, len(horizons)))

    # Initial persistent state
    init_sample = torch.as_tensor(test_in[0:1], dtype=torch.long, device=device)
    state = model.clue_proj(model.embed_tokens(init_sample).view(1, 9, 9, 256))

    batch_size = 32
    for h_idx, k in enumerate(horizons):
        curr_state = state.clone()
        if curr_state.shape[0] != batch_size:
            curr_state = curr_state[0:1].expand(batch_size, -1, -1, -1).clone()

        for s in range(0, num_test, batch_size):
            e = min(s + batch_size, num_test)
            b = e - s
            inp_b = torch.as_tensor(test_in[s:e], dtype=torch.long, device=device)
            lbl_b = torch.as_tensor(test_lbl[s:e], dtype=torch.long, device=device)

            next_state, logits, _ = model.forward_stream_step(curr_state[:b], inp_b, k_step=k)
            preds = torch.argmax(logits, dim=-1)
            accs = (preds == lbl_b).float().mean(dim=-1).cpu().numpy()
            loss = F.cross_entropy(logits.view(-1, 11), lbl_b.view(-1), reduction="none").view(b, 81).mean(dim=-1).cpu().numpy()

            puzzle_accs[s:e, h_idx] = accs
            puzzle_losses[s:e, h_idx] = loss
            curr_state[:b] = next_state.detach()

    # Differential Analysis: Compare Deep Thinking (K=64, 256, 1024) vs Fast Reflex (K=2)
    acc_k2 = puzzle_accs[:, 1]
    acc_k64 = puzzle_accs[:, 6]
    acc_k256 = puzzle_accs[:, 8]
    acc_k1024 = puzzle_accs[:, 9]

    # Best deep performance (max across K >= 16)
    acc_deep_best = np.max(puzzle_accs[:, 4:], axis=1)
    best_k_idx = np.argmax(puzzle_accs, axis=1)
    best_k_vals = np.array([horizons[i] for i in best_k_idx])

    deep_winners = np.where(acc_deep_best > acc_k2 + 1e-4)[0]
    k2_winners = np.where(acc_k2 > acc_deep_best + 1e-4)[0]
    invariants = np.where(np.abs(acc_deep_best - acc_k2) <= 1e-4)[0]

    print("\n" + "=" * 95)
    print("   DISSECTING THE PONDERING PARADOX: PER-PUZZLE COGNITIVE DISTRIBUTION")
    print("=" * 95)
    print(f"Total Test Puzzles: {num_test}")
    print(f"1. Puzzles that GAIN accuracy from deep pondering (Deep Thinking Winners): {len(deep_winners)} ({len(deep_winners)/num_test*100:.1f}%)")
    print(f"2. Puzzles where K=2 is sufficient/identical (Invariant Fixed Points):     {len(invariants)} ({len(invariants)/num_test*100:.1f}%)")
    print(f"3. Puzzles that LOSE accuracy from deep pondering (Degradation Cases):      {len(k2_winners)} ({len(k2_winners)/num_test*100:.1f}%)")

    # Quantify gains on the Deep Winners
    gain_on_winners = (acc_deep_best[deep_winners] - acc_k2[deep_winners]) * 100
    print(f"\nAverage gain on Deep Thinking Winners: +{np.mean(gain_on_winners):.2f}% (Max gain: +{np.max(gain_on_winners):.2f}%)")

    # Hard Puzzles Breakdown (Clues <= 22, representing Extreme Difficulty)
    hard_mask = clue_counts <= 22
    hard_indices = np.where(hard_mask)[0]
    print(f"\n--- Hard Extreme Puzzles (Clues <= 22, Total={len(hard_indices)}) ---")
    print(f"  Mean Accuracy at K= 1:   {np.mean(puzzle_accs[hard_indices, 0])*100:.2f}%")
    print(f"  Mean Accuracy at K= 2:   {np.mean(puzzle_accs[hard_indices, 1])*100:.2f}%")
    print(f"  Mean Accuracy at K=16:   {np.mean(puzzle_accs[hard_indices, 4])*100:.2f}%")
    print(f"  Mean Accuracy at K=64:   {np.mean(puzzle_accs[hard_indices, 6])*100:.2f}%")
    print(f"  Mean Accuracy at K=256:  {np.mean(puzzle_accs[hard_indices, 8])*100:.2f}%")
    print(f"  Mean Accuracy at K=1024: {np.mean(puzzle_accs[hard_indices, 9])*100:.2f}%")

    # Top 5 Individual Puzzles where Deep K massively crushed K=2
    top_diff_idx = deep_winners[np.argsort(-gain_on_winners)[:8]]
    print("\n--- Top 8 Puzzles where Deep Pondering (Large K) Massively Wins ---")
    print(f"{'Idx':^6} | {'Clues':^7} | {'K=1':^8} | {'K=2':^8} | {'K=16':^8} | {'K=64':^8} | {'K=256':^8} | {'K=1024':^8} | {'Net Gain':^9}")
    print("-" * 88)
    for p_idx in top_diff_idx:
        c = clue_counts[p_idx]
        a1 = puzzle_accs[p_idx, 0] * 100
        a2 = puzzle_accs[p_idx, 1] * 100
        a16 = puzzle_accs[p_idx, 4] * 100
        a64 = puzzle_accs[p_idx, 6] * 100
        a256 = puzzle_accs[p_idx, 8] * 100
        a1024 = puzzle_accs[p_idx, 9] * 100
        best_a = max(a16, a64, a256, a1024)
        print(f"{p_idx:5d}  | {c:5d}   | {a1:6.1f}% | {a2:6.1f}% | {a16:6.1f}% | {a64:6.1f}% | {a256:6.1f}% | {a1024:6.1f}% | +{best_a - a2:5.1f}%")


if __name__ == "__main__":
    main()
