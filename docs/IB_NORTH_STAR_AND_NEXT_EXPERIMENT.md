# Information Boltzmann: active line and evidence rule

## North star

Build one persistent kinetic field that can ingest an ordered language stream,
retain useful structure through transport and collision, and expose a generic
reasoning port without replacing the field with a task-specific solver.

The active implementation is `information_boltzmann/`.  Its current research
configuration uses eight complete field paths, a learned physical duration
`T`, and fixed numerical resolution `K=64`.  `K` is integration resolution;
it is never a count of external events.

## Current capability rule

OpenWebText GPT-2 BPE is the primary capability surface.  A language result is
eligible for comparison only when it records:

1. joint optimizer updates and tokens consumed;
2. a saved model, optimizer, stream state, and RNG state from the same update;
3. fixed held-out local sites, 256 observed warm-in tokens, and the following
   128 scored tokens;
4. the exact state-initialization policy; and
5. the checkpoint and source hashes.

The canonical evaluator is `scripts/ib/evaluate_continuous_owt.py`.

## Registered language evidence

The registry at `results/published/ib_language_registry_20260929.json` is the
only source for current language ranking.

| Entry | Mean NLL | Meaning |
| --- | ---: | --- |
| Q8 compatibility replay, 3000 updates | 7.21536 | Current continuous-velocity field control. |
| GDN-2, 3000 updates | 6.49815 | Current matched local-text-window reference. |
| Historical Q8, 3000 updates | 7.19250 | Archive-only observation: its checkpoint is gone and its source hash differs from the current implementation. |

The current Q8 control has positive collision evidence, yet it has no valid
NLL victory over GDN-2.  In particular, a historical Q8 Site 3 score of
6.4409 cannot be compared to a GDN global validation score.  Direct local-site
evaluation gives GDN-2 Site 3 NLL 5.74328.

## Next experiment

Before introducing another mechanism, measure the current Q8 and GDN-2
controls across 16 fixed held-out sites.  In parallel, record boundary
admittance, write-to-field ratio, collision exposure, and collision SNR.  The
current Q8 macro-energy matches the archived run, while those four
signal-bearing quantities are lower.  The next model change must target one
of them and be trained jointly for at least 3000 updates.
