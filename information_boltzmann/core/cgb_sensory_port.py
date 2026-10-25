"""Optional frozen sensor, independent of exact byte generation and brain rules.

Synchronous encoding does not schedule a continuously advancing world. Its owner
declares arrival ticks; wall latency and model/workspace costs stay separate.
"""

from dataclasses import dataclass
from importlib import metadata
from time import perf_counter

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


class SensoryPortViolation(ValueError):
    """Invalid finite input, provenance or causal access."""


@dataclass(frozen=True)
class SensoryReceipt:
    source: str
    revision: str
    modalities: tuple[str, ...]
    source_start: int
    source_end: int
    requested_at: int
    arrival_tick: int
    dimensions: int
    vector: bytes
    input_bytes: int
    latency_seconds: float
    backend: str
    model_storage_bytes: int | None

    @property
    def nbytes(self):
        return len(self.vector)

    def read(self, now):
        if type(now) is not int or now < self.arrival_tick:
            raise SensoryPortViolation("sensor features have not arrived")
        return memoryview(self.vector).cast("f")


class FrozenSensoryPort:
    """Lazy adapter; injected encoders must be pure, frozen and allocation-bounded.
    Decoded media only; UTF-8 measurement temporarily uses at most 4*max_input_bytes.
    Input bounds do not establish a bound on encoder workspace or model quality."""

    def __init__(self, *, encoder=None, dimensions=768, modalities=("text",),
                 revision=REVISION, device="cpu", dtype="float32", allow_download=False,
                 max_input_bytes=2**20, max_items=32):
        if (dimensions not in (128, 256, 512, 768) or dtype not in ("float32", "bfloat16")
                or not isinstance(modalities, tuple) or not modalities
                or len(set(modalities)) != len(modalities)
                or any(m not in ("text", "image", "video", "audio") for m in modalities)
                or not isinstance(revision, str) or not revision
                or (encoder is None and (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)))
                or not isinstance(device, str) or not device
                or type(allow_download) is not bool
                or any(type(x) is not int or x < 1 for x in (max_input_bytes, max_items))):
            raise SensoryPortViolation("invalid sensor configuration; FP16 is unsupported")
        self.encoder, self.dimensions, self.modalities = encoder, dimensions, modalities
        self.revision, self.device, self.dtype = revision, device, dtype
        self.allow_download, self.max_input_bytes, self.max_items = allow_download, max_input_bytes, max_items
        self.backend = "injected" if encoder is not None else MODEL
        self.model = None

    def _measure(self, payload):
        import numpy as np
        count = 0
        def visit(value, media=False, depth=0):
            nonlocal count
            count += 1
            if count > self.max_items or depth > 4:
                raise SensoryPortViolation("input has too many items or nested containers")
            if isinstance(value, str) and not media:
                if len(value) > self.max_input_bytes:
                    raise SensoryPortViolation("input exceeds byte budget")
                return len(value.encode("utf-8"))
            if isinstance(value, np.ndarray) and value.dtype.kind in "buif":
                return value.nbytes
            if type(value).__module__.startswith("torch") and hasattr(value, "numel"):
                return value.numel() * value.element_size()
            if type(value).__module__.startswith("PIL.") and value.mode in ("L", "RGB", "RGBA"):
                return value.width * value.height * len(value.getbands())
            if isinstance(value, (list, tuple)):
                return sum(visit(x, media, depth + 1) for x in value)
            raise SensoryPortViolation("input requires finite text or decoded media")
        if isinstance(payload, dict):
            if set(payload) != set(self.modalities):
                raise SensoryPortViolation("payload modalities differ from configured modalities")
            size = sum(visit(value, key != "text") for key, value in payload.items())
        elif self.modalities == ("text",):
            size = visit(payload)
        else:
            raise SensoryPortViolation("multimodal inputs require an explicit modality dictionary")
        if size > self.max_input_bytes:
            raise SensoryPortViolation("input exceeds byte budget")
        return size

    def _load(self, torch):
        if self.encoder is not None:
            return
        from packaging.version import Version
        try:
            installed = metadata.version("sentence-transformers")
        except metadata.PackageNotFoundError as exc:
            raise SensoryPortViolation("install compatible sentence-transformers >=6.1.0 explicitly") from exc
        if Version(installed) < Version("6.1.0"):
            raise SensoryPortViolation("EmbeddingGemma 2 needs sentence-transformers >=6.1.0")
        from sentence_transformers import SentenceTransformer
        config = {name + "_config": None for name, kind in (("vision", "image"), ("audio", "audio"))
                  if kind not in self.modalities and not (name == "vision" and "video" in self.modalities)}
        self.model = SentenceTransformer(MODEL, revision=self.revision, device=self.device,
            local_files_only=not self.allow_download, config_kwargs=config,
            model_kwargs={"torch_dtype": getattr(torch, self.dtype)})
        self.encoder = lambda payload: self.model.encode(payload, truncate_dim=self.dimensions,
            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)

    def encode(self, payload, *, source, source_start, source_end, observed_through, arrival_tick):
        ticks = (source_start, source_end, observed_through, arrival_tick)
        if (not isinstance(source, str) or not source or any(type(t) is not int or t < 0 for t in ticks)
                or not source_start <= source_end <= observed_through or arrival_tick <= observed_through):
            raise SensoryPortViolation("source is future data or arrival is not after its source")
        size = self._measure(payload)
        import torch
        if self.dtype == "bfloat16" and self.device.startswith("cuda") and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise SensoryPortViolation("native BF16 is unavailable; use FP32")
        start = perf_counter()
        self._load(torch)
        frozen = self.model if self.model is not None else self.encoder
        if hasattr(frozen, "eval"):
            frozen.eval().requires_grad_(False)
        with torch.inference_mode():
            vector = torch.as_tensor(self.encoder(payload)).detach().to(device="cpu", dtype=torch.float32)
            if vector.shape not in ((self.dimensions,), (1, self.dimensions)) or not bool(torch.isfinite(vector).all()):
                raise SensoryPortViolation("encoder returned invalid shape or nonfinite features")
            vector = vector.reshape(-1).contiguous()
            norm = float(vector.double().norm())
            if norm == 0:
                raise SensoryPortViolation("encoder returned a zero vector")
            storage = (vector.double() / norm).float().numpy().tobytes()
        model_bytes = None
        if hasattr(frozen, "parameters"):
            stores = {(str(p.device), p.untyped_storage().data_ptr()): p.untyped_storage().nbytes()
                      for p in (*frozen.parameters(), *frozen.buffers())}
            model_bytes = sum(stores.values())
        return SensoryReceipt(source, self.revision, self.modalities, source_start, source_end,
            observed_through, arrival_tick, self.dimensions, storage, size, perf_counter() - start, self.backend, model_bytes)
