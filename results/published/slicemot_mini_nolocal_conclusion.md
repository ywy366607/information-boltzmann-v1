# SliceMoT-Mini-2L arm=`nolocal`

- note: base + LocalVisual OFF (identity after Deslice)
- layers=2 M=32 d=256 d_x=256 (aligned)
- Gemma + LoRA r=8 last-4; layer Slice params **independent**
- flags: ada_temp=True gumbel=False topk=0 stiefel=True local=none
- Native MoT only (**no Patch**)

## 1× token gate
- gate_params (stack+LoRA) = **4807347**
- loss tokens needed (≥1.0×) = **4807347**
- loss tokens seen = **1204654** (ratio=0.251)
- **eligible_1x = False**
- steps=100000 final_loss=1.4701 VRAM=1949MB

## Lite visual-causality probes
- matched NLL = 2.5376
- shuffled-image NLL = 2.8158
- Δ (shuffle−matched) = +0.2782
- frac matched lower NLL = 0.771
- TF caption-span acc = 0.426

## Decision
- **No efficacy claim**: did not reach 1× token budget.

Knob attribution: Ada-Temp/Gumbel = Transolver++; topk/Stiefel = this repo; T3 scale stack not in scope.
Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
