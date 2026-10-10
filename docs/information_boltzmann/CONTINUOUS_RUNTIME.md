# Continuous observation, evolution and readout

This runtime puts the plastic medium on one persistent clock. Observations are
timestamped boundary events, not a request to finish a fixed pondering duration.
The medium evolves between observations, including input-free intervals. Readout
observes the latest committed state and can happen while the stream continues.

## 1. State and time

The state is `(field, flux_x, flux_y, flux_z, conduction, precision, elapsed)`.
Every continuation checkpoint retains all of these tensors. The elapsed clock is
FP64 even when the physical fields use FP32; it does not lose small intervals
simply because the individual has run for a long time.

For observation times `o_i` and read times `r_i`, the next-token supervised path is

```
evolve to o_i -> assimilate observed token -> evolve to r_i -> read
```

with `o_i <= r_i < o_(i+1)`. The next observation is absent from its own prediction
path. Writing and reading consume no *model* time; their wall-clock cost is real.
The medium has no condition that evolution must settle before an output is read.

`max_step` is a declared numerical accuracy budget, in model-time units. It splits
an elapsed interval into solver steps; it is not a token-specific cognitive-time
policy. Accuracy is assessed by reducing it at fixed event times. The existing
structural relaxation and quadratic bath retain their analytic frozen-step
updates. The nonlinear complete flow still needs refinement checks.

## 2. Differentiable training engine

`ContinuousStream` is synchronous and differentiable. It carries gradients
through timestamped evolution and writes without implicit detach. The model's
`forward_timestamped` uses exactly this event engine, batches the decoder GEMM,
and keeps the current next-token CE plus write-port objective:

```python
loss, belief, metrics = model.forward_timestamped(
    input_ids, next_ids, observation_times, read_times,
    max_step=solver_accuracy_budget, belief=belief,
)
loss.backward()
optimizer.step()
belief = belief.detach()  # explicitly end the chosen BPTT window; keep the state
```

The caller supplies timestamps appropriate to the data and deployment cadence.
Training on reads at several causal offsets can supervise outputs during the
flow. This infrastructure preserves ordinary BPTT credit assignment; longer
credit assignment remains a separate learning question. No corpus training was
started as part of this runtime change.

## 3. Background deployment owner

`LiveStream` runs one state owner in a background thread. Producers submit
observations or read requests and receive `concurrent.futures.Future` immediately.
The owner serializes physical updates, so action and perception producers do not
mutate the same tensor concurrently.

```python
from information_boltzmann.runtime import ContinuousStream, LiveStream

model.eval()
stream = ContinuousStream(model, max_step=solver_accuracy_budget, belief=belief)
live = LiveStream(stream, model_time_per_second=physical_clock_scale).start()
observed = live.submit_observation(token_id_cpu)  # timestamp from the live clock
response_future = live.request_read()             # latest state at next safe point
response = response_future.result()
print(response.state_time, response.lag)
belief = live.close()                             # retain state for continuation
```

The wall/model-time scale is explicit; assigning biological milliseconds requires
physical calibration. The live owner uses fixed model weights, while conduction
continues local autonomous adaptation. Joint gradient learning uses the training
engine above. An optimizer must not mutate model weights concurrently with the
deployment owner.

* Current-state reads take priority at the next complete solver boundary, including
  while the owner is catching up after a long gap. Their response reports its
  actual state timestamp and lag from the request timestamp.
* An explicitly timestamped read waits until that time. Events at equal times
  preserve submission order. No event observes future inputs.
* The bounded queue reports overload through `queue.Full`. Late observations
  report an error instead of silently rewinding the individual. Worker failures
  resolve both queued and currently executing futures.
* On CUDA the owner synchronizes before publishing results. This keeps producer
  submission nonblocking while ensuring returned snapshots are complete.

Compute throughput must exceed the chosen physical cadence for real-time catch-up.
The exposed lag makes overload visible. Background scheduling removes a forced
per-token waiting interface; it does not make numerical computation free.

## 4. Execution changes

Fixed-weight inference caches material coefficients, baseline conduction,
plasticity coefficients and the normalized embedding table. Parameter-version
changes invalidate caches; explicit `.data` edits require `invalidate_cache()`.
Dynamic field, conduction and precision remain actual state, updated on every
applicable event. Training builds coefficient graphs within each interval and
does not reuse detached inference caches.

Ordinary deployment evolution skips diagnostic energy/variance reductions. The
same physical operations run, and opt-in monitoring can use the original path.
This preserves the original forward equations rather than reducing the dynamics.

The `cuda_graph` backend implements a fixed-shape deployment graph with dynamic
state and duration inputs. It owns output copies, and material changes cause
recapture. The deployment graph rejects gradients. Differentiable training uses
the standard kernel, `torch.compile`, or the separate forward/backward training
graph in `runtime/training.py`; see [CONDUCTANCE_TRAINING_EXECUTION.md](CONDUCTANCE_TRAINING_EXECUTION.md).

## 5. Verification and measured execution

49 numerical/interface/regression tests passed, covering causal timestamps,
training gradients, explicit detach, continuation, cache invalidation, live idle
evolution, nonblocking submission, urgent reads during catch-up and failure
propagation. Two opt-in CUDA checks are skipped under the current guard.

On CPU, one thread, actual 8x8x4/D128 dynamics, 32 measured ticks after warm-up:

| Path | ms/tick | Maximum state difference |
| --- | ---: | ---: |
| Reference with diagnostics | 10.651 | 0 |
| Cached coefficients, quiet execution | 9.802 | 0 |

The matched gain is 1.087x. This is a dynamics execution benchmark, not a language
training-throughput or reasoning-capability result.

The initial GPU benchmark was allowed only with zero additional Windows shared-memory
usage above CUDA initialization. With a 12% CUDA allocator limit, WDDM reported
76 MiB shared at initialization, then 78 MiB after the reference variant. The
guard stopped before graph capture. This counter alone cannot identify whether
the 2 MiB is driver workspace or VRAM paging; it does establish measured growth,
so no zero-growth CUDA result or graph speedup is claimed. The independent fly
training was neither stopped nor restarted.

Artifacts: `results/published/continuous_execution_cpu.json` and
`results/published/continuous_execution_gpu.json`. Reproduce the CPU path with

```powershell
python scripts/ib/benchmark_continuous_execution.py --device cpu --step-duration 0.005 --ticks 32 --output results/published/continuous_execution_cpu.json
```

The declared duration is a numerical test setting, not a biological frequency.
Enable CUDA tests with `IB_ENABLE_RUNTIME_CUDA_TESTS=1` within an explicitly
available memory budget. On 2026-10-03 the user revised the requirement: monitor
total dedicated GPU memory and keep it below 4 GiB; shared-counter growth is
permitted. Both execution benchmarks implement this policy by default.
`--require-zero-shared-growth` in the dynamics benchmark reproduces the old
policy. Shared counters remain recorded independently of dedicated memory.

Under the revised requirement, all five opt-in deployment/training CUDA checks
pass, including variable-duration continuation, immutable returned snapshots,
material-triggered recapture, two-chunk gradient accumulation, and parameter
updates without training recapture. The historical stopped report remains an
account of the original guard, rather than an execution-readiness blocker.

The earlier full W4 boundary-action/energy closure gate remains recorded in
`PLASTIC_MEDIUM_FEASIBILITY.md`. Scheduling and kernel equivalence establish
execution readiness; source-policy closure and sufficient joint real-data
training establish the learning candidate's remaining readiness.

The conductance response upgrade adds persistent `receptors` to this state and
runtime schema2. See [CONDUCTANCE_RESPONSE.md](CONDUCTANCE_RESPONSE.md) for its
learned physical coefficients, source-work accounting and separate certificate.
Historical schema1 quadratic continuations remain supported. A conductance
continuation requires its saved receptors; missing state is an explicit error.
