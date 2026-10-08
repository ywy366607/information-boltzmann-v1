"""Isolated captured vocabulary forward/backward precision comparison.

This reports GEMM cost only, not an estimated production update or capability.
The FP16 loss scale is calibrated for finite gradients on this fixed input.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    table_value = F.normalize(saved["model"]["source.embedding.weight"], dim=-1).detach()
    summary = saved["state"].mean((1, 2, 3))
    prior = torch.cat((F.rms_norm(summary, (table_value.shape[1],)), saved["precision"].log()), -1)
    weights = saved["model"]
    natural_value = F.linear(F.silu(F.linear(
        prior, weights["write_agent.port_prior.0.weight"], weights["write_agent.port_prior.0.bias"])),
        weights["write_agent.port_prior.2.weight"], weights["write_agent.port_prior.2.bias"]).detach()
    bias_value = weights["write_agent.port_logit_bias"].detach()
    del saved
    reference_grad = None
    rows = []
    for dtype in (torch.float32, torch.float16):
        table = table_value.clone().requires_grad_()
        natural = natural_value.clone().requires_grad_()
        bias = bias_value.clone().requires_grad_()
        target = torch.tensor([1234], device="cuda")

        def forward():
            vocabulary = table.to(dtype)
            logits = (natural.to(dtype) @ vocabulary.T).float() + bias
            probability = logits.softmax(-1)
            expected = (probability.to(dtype) @ vocabulary).float()
            loss = F.cross_entropy(logits, target) + expected.square().sum()
            return loss

        # No model update occurs: find a usable numeric scale for the fixed
        # derivative before measuring the half-precision arithmetic.
        scale = 1.0 if dtype == torch.float32 else 65536.0
        while True:
            for parameter in (table, natural, bias):
                parameter.grad = None
            (forward() * scale).backward()
            if all(torch.isfinite(parameter.grad).all() for parameter in (table, natural, bias)):
                break
            scale /= 2
            if scale < 1:
                raise FloatingPointError("No finite FP16 derivative on benchmark input")
        grad = torch.cat([p.grad.flatten() / scale for p in (table, natural, bias)])
        relative_error = 0.0 if reference_grad is None else float((grad - reference_grad).norm() / reference_grad.norm())
        if reference_grad is None:
            reference_grad = grad.clone()

        compiled_forward = torch.compile(forward)

        def backward():
            for parameter in (table, natural, bias):
                parameter.grad.zero_()
            loss = compiled_forward()
            (loss * scale).backward()
            return loss

        # Keep the same launch-fusion machinery as the production writer.
        run = backward
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = run()
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 10)
        row = {"dtype": str(dtype), "median_ms": statistics.median(times),
               "loss": float(loss.detach()), "loss_scale": scale,
               "relative_gradient_error": relative_error}
        print(json.dumps(row), flush=True)
        rows.append(row)
        del graph, run, compiled_forward
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "isolated_two_vocabulary_GEMMs_plus_probability_and_backward",
                                       "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
