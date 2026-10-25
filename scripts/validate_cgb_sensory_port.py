"""Offline sensor-interface checks with injected encoders; no real-model inference."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
path = root / "information_boltzmann/core/cgb_sensory_port.py"
spec = importlib.util.spec_from_file_location("cgb_sensory_port", path)
sensor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sensor
before = set(sys.modules)
spec.loader.exec_module(sensor)
assert "torch" not in set(sys.modules) - before
assert "sentence_transformers" not in set(sys.modules) - before

import numpy as np
import torch


class Encoder(torch.nn.Module):
    def __init__(self, dimensions=768):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.dimensions = dimensions
        self.calls = 0

    def forward(self, payload):
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert not self.training and not self.weight.requires_grad
        self.calls += 1
        return self.weight * torch.arange(1, self.dimensions + 1, dtype=torch.float32)


def reject(operation, fragment=""):
    try:
        operation()
    except (sensor.SensoryPortViolation, ValueError) as exc:
        assert fragment in str(exc)
    else:
        raise AssertionError("invalid sensor operation succeeded")


def encode(port, payload="already arrived", **changes):
    args = dict(source="recorded-input", source_start=2, source_end=4,
                observed_through=4, arrival_tick=5)
    args.update(changes)
    return port.encode(payload, **args)


def main():
    for dimensions in (128, 256, 512, 768):
        encoder = Encoder(dimensions)
        port = sensor.FrozenSensoryPort(encoder=encoder, dimensions=dimensions)
        receipt = encode(port)
        assert receipt.backend == "injected" and receipt.revision == sensor.REVISION
        assert receipt.nbytes == dimensions * 4 and receipt.input_bytes == 15
        assert receipt.latency_seconds >= 0 and receipt.model_storage_bytes == 4
        reject(lambda: receipt.read(4), "not arrived")
        view = receipt.read(5)
        assert view.readonly and view.itemsize == 4 and view.format == "f"
        vector = np.frombuffer(receipt.vector, dtype=np.float32)
        expected = np.arange(1, dimensions + 1, dtype=np.float64)
        expected /= np.linalg.norm(expected)
        assert np.allclose(vector, expected) and np.allclose(vector, list(view))
        assert not encoder.training and not encoder.weight.requires_grad
        assert encoder.weight.grad is None and not hasattr(port, "history")
        calls = encoder.calls
        reject(lambda: encode(port, source_end=5), "future")
        reject(lambda: encode(port, arrival_tick=4), "arrival")
        reject(lambda: encode(port, observed_through=100), "arrival")
        assert receipt.requested_at == 4 and receipt.arrival_tick == 5
        assert encoder.calls == calls
    reject(lambda: sensor.FrozenSensoryPort(dtype="float16"), "FP16")
    reject(lambda: sensor.FrozenSensoryPort(device=object()), "configuration")
    reject(lambda: sensor.FrozenSensoryPort(dimensions=767), "configuration")
    limited = sensor.FrozenSensoryPort(encoder=Encoder(), max_input_bytes=3)
    reject(lambda: encode(limited, "four"), "byte budget")
    class UnencodedText(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("oversized text was encoded before rejection")
    reject(lambda: encode(limited, UnencodedText("x" * 1000)), "byte budget")
    assert limited.encoder.calls == 0
    nested = sensor.FrozenSensoryPort(encoder=Encoder(), max_items=2)
    reject(lambda: encode(nested, ["a", "b"]), "too many")
    media = sensor.FrozenSensoryPort(encoder=Encoder(), modalities=("video",))
    reject(lambda: encode(media, {"video": "future/full.mp4"}), "decoded media")
    payload = {"video": np.zeros((2, 8, 8, 3), dtype=np.uint8)}
    assert encode(media, payload).input_bytes == 384
    reject(lambda: encode(media, {"image": payload["video"]}), "modalities")
    for output in (torch.zeros(768), torch.full((768,), float("nan")), torch.ones(767)):
        broken = sensor.FrozenSensoryPort(encoder=lambda _: output)
        reject(lambda: encode(broken), "encoder returned")
    huge = sensor.FrozenSensoryPort(encoder=lambda _: torch.full((768,), 3e38))
    assert np.isfinite(np.frombuffer(encode(huge).vector, dtype=np.float32)).all()
    with patch.object(sensor.metadata, "version", return_value="6.0.0"):
        reject(lambda: encode(sensor.FrozenSensoryPort()), ">=6.1.0")
    captured = {}
    class FakeSentenceTransformer(Encoder):
        def __init__(self, model, **kwargs):
            super().__init__()
            captured.update(model=model, **kwargs)
        def encode(self, payload, **kwargs):
            assert kwargs["normalize_embeddings"] and kwargs["convert_to_tensor"]
            return self(payload)[:kwargs["truncate_dim"]]
    fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    with patch.object(sensor.metadata, "version", return_value="6.1.0"):
        with patch.dict(sys.modules, sentence_transformers=fake_module):
            receipt = encode(sensor.FrozenSensoryPort())
    assert captured["model"] == sensor.MODEL and captured["revision"] == sensor.REVISION
    assert captured["local_files_only"] is True and captured["device"] == "cpu"
    assert captured["model_kwargs"]["torch_dtype"] == torch.float32
    assert captured["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert receipt.backend == sensor.MODEL
    print("cgb_sensory_port_ok; injected/fake backend only; no downloads or model-quality claim")


if __name__ == "__main__":
    main()
