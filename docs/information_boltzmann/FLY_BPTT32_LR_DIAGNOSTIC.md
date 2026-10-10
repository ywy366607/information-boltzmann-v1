# Fly BPTT32 AdamW learning-rate and gradient diagnosis

Date: 2026-10-05

The production learner was saved and briefly paused at fresh training cursor 234464, physical event 258040 and optimizer update 4514. The diagnostic uses the complete saved life, trained weights and both mature AdamW optimizer states. Production was then resumed from that exact checkpoint, preserving target 200000 cumulative BPTT training targets and LR 0.0002.

## Paired protocol

Three consecutive 32-target real OWT windows from the saved training cursor. The reference physical life rolls forward under the original checkpoint weights. Within each window, every alternative restores the identical original weights, Adam moments, current clipped surrogate gradient and full initial physical state. Each candidate is an actual native AdamW update, including existing weight decay and edge bounds. Parameters are never updated in production by the diagnostic.

Fitting-window loss is recomputed from the same pre-window physical state. Next-window loss starts from the same original-weight post-window physical state; this matches the actual learner, whose state was computed before the optimizer update. This is a local conditional optimizer diagnostic, not primary active evaluation, training, convergence evidence or a claim about long-run retention. The spikes retain hard forward thresholds and ATan surrogate backward; disagreements between the first-order estimate and finite response can also reflect that approximation.

## Finite-step results

| Alternative | Mean fitting-window NLL reduction | Mean next-window NLL reduction |
| --- | ---: | ---: |
| half | 0.00997448 | 0.00164207 |
| current | 0.01984819 | 0.00334613 |
| double | 0.03931300 | 0.00673962 |
| writer_only | 0.00012779 | 0.00003497 |
| synapse_only | 0.00028483 | 0.00003846 |
| read_decoder_only | 0.01940823 | 0.00324106 |
| physiology_only | 0.00001907 | -0.00003147 |

All three global learning-rate alternatives improve both metrics in each of the three windows. Double LR yields roughly double local benefit, with no observed local overshoot. The current 0.0002 stays unchanged: long-run speed, accuracy and retention are assessed by the continuing learner.

Only updating read/decoder provides about 97.8% of the full immediate fitting-window improvement. This is local update attribution; it does not measure the cumulative contribution of the brain dynamics.

## Raw gradient scale

| Group | Parameters | Mean L2 norm | Per-coordinate RMS | Active fraction |
| --- | ---: | ---: | ---: | ---: |
| physiology | 164 | 0.0702309 | 0.00548411 | 84.756% |
| synapse | 25321395 | 0.0218424 | 4.34067e-06 | 12.065% |
| writer | 12222723 | 0.00839379 | 2.4009e-06 | 98.410% |
| read_decoder | 40440145 | 0.927758 | 0.000145891 | 99.987% |

Historical writer_grad_norm covers only the three sensory projections. Writer gates now have a separate monitor and writer_total_grad_norm includes both. Output projection, vocabulary decoder and read RMSNorm gain are also monitored separately. Norm differences persist after dividing by sqrt(parameter count). AdamW normalizes coordinates using gradient moments, so equal raw gradient norms are not required for effective learning.

Read RMSNorm gain accounts for 61.2% of raw squared-gradient mass, decoder weight for 33.9%, decoder bias for 4.0%, and output projection for 0.235% (averages of per-window shares). A large combined read/decoder norm should not be confused with a large motor-to-readout projection gradient. The current edge relative updates average 0.308% E and 0.262% I per diagnostic step, while their immediate NLL effect is much smaller than the decoder effect.

Expanded monitoring adds writer_gate_grad_norm, writer_total_grad_norm, read_grad_norm, decoder_grad_norm, read_norm_grad_norm and per-coordinate RMS metrics. No group-wise gradient equalization or learning-rate changes were applied.

## Reproduction

```powershell
D:\conda_envs\vox\python.exe scripts/ib/diagnose_fly_bptt_learning_rate.py --checkpoint results/q8_fly_bptt32_adamw_continuous_100k/last.pt --windows 3
```

The raw report identifies the saved cursor and moments; the production last.pt continues changing after resumption. Six affected numerical/interface tests pass. Diagnostic peak allocated 2203 MiB and reserved 3084 MiB; production is running again around 0.684 s / 32 events.
