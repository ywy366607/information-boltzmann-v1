# Persistent fly: decoder half-rate, 30M fresh-target continuation

Date: 2026-10-05

Status update: this one-tick timing run was saved and stopped at the user's
request, at 600,928 BPTT training targets / train cursor 700,928. The original
launch/recovery instructions below are retained as historical provenance. The
prepared repaired timing and its exact continuation command are documented in
[FLY_PULSE_QUIET_TIMING.md](FLY_PULSE_QUIET_TIMING.md); long training is currently
stopped and the saved 30M target remains unchanged.

The user authorizes at least 30 million additional non-repeated training tokens. Continue the same physical individual from training cursor 282144, BPTT training count 182144, with all weights, Adam moments and COBA/ALIF/STP state preserved. Static sensory input and motor output ports, BPTT32, frozen embedding and STP parameters are unchanged.

## Optimizer

Decoder **weight** LR is 0.0001. Decoder bias, read projection, read RMSNorm, writer, synapses and physical parameters use 0.0002. All currently trainable groups use AdamW. Decoder regrouping migrates moments by parameter name, including old two-group checkpoints; seven numerical/interface tests pass. The split does not reset the optimizer step, momentum, physical state or data cursor.

## Data budget

Prepared data/ib_owt_gpt2_31m contains 31,000,387 train targets, 371,722 validation targets and 325,029 test targets, from pinned OpenWebText revision 433fe0f44ed7894fea29c08b3202aa348ccc6369. Existing train/validation/test prefixes are byte-equivalent as token arrays. Additional documents use the same NFC/SHA256 split and GPT-2 tokenizer, deduplicated against all existing and new documents. Each fresh training index is consumed once; the cursor never wraps. Normal recurrence of vocabulary words is expected, while corpus/document replay is excluded from the fresh training budget.

Fresh training budget: **30,000,000** targets and **937,500** BPTT32 optimizer updates. Target cumulative BPTT count: **30,182,144**. Active validation and intentional A-to-B-to-A revisit learning are counted separately and are excluded from these 30M fresh training targets. Evaluation uses the actual learner, pre-update scores, full state persistence and fresh validation cursors. Cadence is every 100,000 cumulative training targets, plus the final boundary. Prepared validation capacity is checked before capture.

## Run and recovery

PID at launch: 28228. Output retains its historical directory name results/q8_fly_bptt32_adamw_continuous_100k to reuse rolling checkpoints without duplicating a full optimizer checkpoint on the nearly full disk. Fresh-budget progress fields unambiguously show 30M. The previous optimizer segment config and progress are preserved under segments/uniform_adamw_100k. Logs are long30m.stdout.log / long30m.stderr.log. Rolling last.pt includes the complete life; best.pt is a weights-only historical live-validation best.

Initial extension uses --extend-budget exactly once. Recovery uses the command below **without** --extend-budget, retaining the saved target rather than adding another 30M. No further epochs or wrapping are enabled.

```powershell
D:\conda_envs\vox\python.exe -u scripts/ib/train_fly_bptt_stream.py --resume results/q8_fly_bptt32_adamw_continuous_100k/last.pt --output results/q8_fly_bptt32_adamw_continuous_100k --data data/ib_owt_gpt2_31m --additional-tokens 30000000 --window 32 --lr 0.0002 --lr-decoder 0.0001 --lr-synapse 0.0002 --lr-sensory 0.0002 --plasticity-optimizer adamw --validate-every-tokens 100000 --log-every-tokens 1024 --vram-limit-mib 3900
```

Production launch is confirmed at about 0.684 seconds per 32-target window, device-wide memory 2539 MiB (4 GiB GPU). At approximately 43–47 targets/s, the training budget takes about 7.4–8.1 days before interruptions. Learning-rate and capability conclusions will use the learning curve and continuing evaluation; successful launch verifies execution only.
