import sys, os, time
sys.path.insert(0, os.path.abspath("."))
import torch
from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

def test_cuda_engine():
    print("Testing OWTConstructalEngine on CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    engine = OWTConstructalEngine(
        vocab_size=50257,
        dim=64,
        init_slots=128,
        max_slots=256,
        device=device
    )

    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    t0 = time.time()
    for step in range(500):
        c_tok = (step * 37) % 50257
        n_tok = (step * 37 + 13) % 50257
        metrics = engine.step_stream(c_tok, n_tok)
        if step % 100 == 0:
            print(f"Step {step}: NLL={metrics['surprisal_nats']:.3f}, slots={int(metrics['num_slots'])}, beta={metrics['beta_star']:.2f}")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"500 steps completed in {elapsed:.3f}s ({elapsed/500*1000:.2f} ms/step)")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

if __name__ == "__main__":
    test_cuda_engine()
