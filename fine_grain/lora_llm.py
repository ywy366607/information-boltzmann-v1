"""Optional LoRA on frozen causal LMs (Phase 3: trainable language side).

Uses ``peft`` when available. Targets last-N transformer layers' q/v projections
for ~4GB-friendly adapter training next to a vision frontend.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


def peft_available() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def count_transformer_layers(model) -> int:
    """Best-effort layer count for Gemma/LLaMA-style or GPT-2 ``h`` stacks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    # PeftModel wraps base
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        inner = base.model
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return len(inner.model.layers)
        if hasattr(inner, "layers"):
            return len(inner.layers)
    return 0


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    last_n_layers: int = 4,
) -> Tuple[Any, dict]:
    """Wrap ``model`` with LoRA on the last ``last_n_layers`` blocks.

    Base weights stay frozen; only LoRA matrices require grad.
    Returns ``(peft_model, meta)``.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = ("q_proj", "v_proj")
    n_layers = count_transformer_layers(model)
    if n_layers <= 0:
        # fall back: all matching modules (still small with r=8 q/v only)
        layers_to_transform = None
        layer_range = "all"
    else:
        n = max(1, min(int(last_n_layers), n_layers))
        start = n_layers - n
        layers_to_transform = list(range(start, n_layers))
        layer_range = f"{start}:{n_layers}"

    cfg_kw = dict(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if layers_to_transform is not None:
        cfg_kw["layers_to_transform"] = layers_to_transform
    cfg = LoraConfig(**cfg_kw)

    # ensure base frozen before inject
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in peft_model.parameters())
    meta = {
        "lora_r": int(r),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "target_modules": list(target_modules),
        "last_n_layers": int(last_n_layers),
        "layer_range": layer_range,
        "n_layers": int(n_layers),
        "trainable_params": int(n_train),
        "all_params": int(n_all),
        "trainable_pct": float(100.0 * n_train / max(n_all, 1)),
    }
    return peft_model, meta


def enable_lora_grads(model) -> int:
    """Re-enable requires_grad on LoRA params (after bridges that freeze all LLM)."""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            n += p.numel()
    return n


def freeze_non_lora(model) -> None:
    """Freeze everything except LoRA adapter weights."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
