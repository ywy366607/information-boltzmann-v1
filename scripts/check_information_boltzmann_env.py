"""Read-only dependency and tensor preflight; does not train or download data."""

import importlib.metadata
import json
import sys

import torch


def main() -> None:
    packages = ["torch", "numpy", "pytest", "datasets", "tokenizers"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    x.square().sum().backward()
    assert torch.allclose(x.grad, 2 * x.detach())
    print(json.dumps({
        "python": sys.version,
        "executable": sys.executable,
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tensor_backward": "passed",
        "prototype_implemented": False,
    }, indent=2))


if __name__ == "__main__":
    main()
