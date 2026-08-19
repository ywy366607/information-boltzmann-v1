# Native MoT architecture (implemented)

Source of truth: `fine_grain/native_mot.py`

## State per layer

| Symbol | Shape | Role |
|--------|-------|------|
| \(X_l\) | \(N \times d_x\) (default \(d_x=128\)) | Full-res visual point field |
| \(S_l\) | \(M \times d\) (default \(d=512\)) | **Ephemeral** visual slices (layer-local) |
| \(H_l\) | \(T \times d\) | Language state |

## Layer

```
S = SliceRead(X)
S', H' = MoTBlock(S, H)   # private QKV/FFN; shared K/V space only
X' = LocalVisual(X + Deslice(S'))
```

### MoTBlock

1. **Private projections** (not shared across modalities):
   - vision: `RMSNorm_v`, `Wq_v, Wk_v, Wv_v, Wo_v`, `FFN_v`, residual scale
   - language: `RMSNorm_t`, `Wq_t, Wk_t, Wv_t, Wo_t`, `FFN_t`, residual scale
2. **Shared attention space** (not parameters):
   - `K = [Kv; Kt]`, `V = [Vv; Vt]`
   - `Av = softmax(Qv K^T / √d) V`, `At = softmax(Qt K^T / √d) V`
3. **Private experts** on residuals + FFN

Routing is by modality membership (no learned MoE router / load-balance loss).

## Pretrain validation gate

```text
data_tokens ≥ 1 × n_parameters   → eligible
```

Smoke runs on Flickr8k document `validation_eligible_1x=false` until token scale is met.
Scale data: Flickr30k / COCO shards / LLaVA-Pretrain 558k.

## Train

```bash
python scripts/train_native_mot_hf.py --joint --res 32 --d 256 --d_x 128 --n_layers 2 --n_slices 32 --steps 400
# full-width 512 (heavier):
python scripts/train_native_mot_hf.py --joint --d 512 --n_layers 4 --res 64 --steps 2000 --max_rows 6000
```

## Tests

`python tests/test_native_mot.py` — private weights, cross-modal dependence, field pipeline, grads.
