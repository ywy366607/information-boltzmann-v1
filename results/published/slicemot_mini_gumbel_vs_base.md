# SliceMoT mini: Gumbel vs base (ST+Ada-Temp)

## Default recipe (locked)

- **Stiefel ON** — anti-collapse (primary)
- **Ada-Temp ON** — mild assignment temperature
- **Gumbel OFF** — default after this study
- deslice soft or topk ablated separately

## Results summary

| arm | steps | tokens (approx) | train quality | lite causality |
|-----|------:|----------------:|---------------|----------------|
| **base** | 100000 | 1.21M (0.24×) | stable CE ~2–4 | **Δ=+0.25**, frac=0.79, TF=0.42 |
| **gumbel** | 5300 (stop) | 64k (0.013×) | unstable, EMA~8, many spikes | not run |

## Conclusion

**Gumbel is not necessary** when Stiefel is on. Default off. Proceed with **topk2** under the ST+Ada-Temp base.
