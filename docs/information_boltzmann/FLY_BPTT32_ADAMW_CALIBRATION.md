# Fly BPTT32: uniform AdamW update-scale calibration

Date: 2026-10-05

All currently trainable groups now default to AdamW with learning rate 0.0002. Existing readout/decoder/physical Adam moments are retained; previously SGD-trained sensory projections and excitatory/inhibitory edges begin with zero Adam moments. Existing weight-decay policies and clipping are preserved. Embedding and STP parameters remain fixed as in the source checkpoint.

The calibration continues the completed individual from training cursor 200000 and physical event 221272 for eight 32-event updates on fresh real OWT. Full physical state persists, sensory-only input and motor-only readout are preserved. It produces reports only; production checkpoints remain unchanged.

## Measured updates

Relative displacement is ||new - old||_2 / ||old||_2 over each complete parameter tensor.

| Parameter group | First update | Eight-update cumulative displacement |
| --- | ---: | ---: |
| edge_weight_e | 1.004635% | 4.330141% |
| edge_weight_i | 0.271369% | 1.069920% |
| topographic_writer.proj_vis.weight | 0.000507% | 0.001626% |
| topographic_writer.proj_chemo.weight | 0.038861% | 0.156042% |
| topographic_writer.proj_mech.weight | 0.059407% | 0.258111% |
| output_read.weight | 0.054794% | 0.397202% |

For the first actual clipped gradient, the old SGD rule would leave 80.748% of nonzero-gradient excitatory coordinates, 95.277% of inhibitory coordinates and 99.921% of chemical-projection coordinates unchanged in float32. This counterfactual includes the original edge bound and exact PyTorch add_ arithmetic. AdamW makes these small gradients representable as weight updates; equal learning rates still produce different relative tensor displacements.

Median training time after the first update: 0.6817 s / 32 events (46.94 tokens/s). Peak allocated memory: 2222.22 MiB; reserved: 3000 MiB. Timing excludes capture and measurement overhead. Memory includes two parameter snapshots used only for calibration.

All eight losses and gradients are finite. These measurements establish actual parameter updates and resource feasibility. NLL improvement and long-run stability require a full continuation with the changed optimizer; this calibration makes no capability claim. Edge Adam moments are new, so early relative updates include their initialization transient.

## Reproduction

```powershell
D:\conda_envs\vox\python.exe scripts/ib/measure_fly_bptt_update_scale.py --updates 8 --lr 0.0002
D:\conda_envs\vox\python.exe -m pytest tests/test_fly_bptt_learning.py tests/test_fly_bptt_credit.py -q
```

Focused validation: 6 tests passed. The continuation preset is configs/information_boltzmann/fly_bptt32_continuous.json. No long training was launched for this request.
