# SliceMoT-Mini-2L arm=`base`

- note: DEFAULT: Stiefel+Ada-Temp; soft deslice; no Gumbel
- layers=2 M=32 d=256 d_x=256 (aligned)
- Gemma + LoRA r=8 last-4; layer Slice params **independent**
- flags: ada_temp=True gumbel=False topk=0 stiefel=True
- Native MoT only (**no Patch**)

## 1× token gate
- gate_params (stack+LoRA) = **4944051**
- loss tokens needed (≥1.0×) = **4944051**
- loss tokens seen = **1205716** (ratio=0.244)
- **eligible_1x = False**
- steps=100000 final_loss=3.7748 VRAM=1958MB

## Lite visual-causality probes
- matched NLL = 2.8870
- shuffled-image NLL = 3.1361
- Δ (shuffle−matched) = +0.2491
- frac matched lower NLL = 0.792
- TF caption-span acc = 0.419

## Decision
- **No efficacy claim**: did not reach 1× token budget.

Knob attribution: Ada-Temp/Gumbel = Transolver++; topk/Stiefel = this repo; T3 scale stack not in scope.
Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
