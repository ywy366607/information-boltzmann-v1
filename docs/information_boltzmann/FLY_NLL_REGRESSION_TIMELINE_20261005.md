# Fly NLL regression: historical onset and learning-contract changes

## Finding

The first documented deterioration in the current continuous learning lineage
starts around 35k training targets and becomes pronounced at 70k. It precedes
the BPTT32 continuation, quiet-tick repair, input-clock repair and STP activation.
The current 7.948 reading was a phase first-pass training average, not a new
independent validation ceiling. Current STP continuation's first fresh active B
at 1.2M training targets is 7.614604 versus fixed unigram 7.355176.

## Historical evidence

The timeline artifact preserves all encounter rows and references calculated
from the exact scored fresh B tokens using the fixed train-only unigram.

| Continuous stage | Active B mean NLL | Same-text unigram | Excess NLL |
| --- | ---: | ---: | ---: |
| Three-factor online, 5k–30k, six encounters | 7.419410 | 7.317102 | +0.102308 |
| Destination local eligibility, 35k–100k, 14 encounters | 8.081703 | 7.494094 | +0.587609 |
| First BPTT32 continuation, 20 encounters | 7.984413 | 7.392208 | +0.592204 |

The first two groups differ by 0.662293 raw NLL, of which 0.176992 is
unigram-reference change; excess risk increases 0.485301. This establishes
deterioration beyond marginal text difficulty. Conditional text difficulty,
ongoing physical history and learner changes remain mixed in that excess.

The 15k best of 7.135239 is a real active B result; its same-text reference is
6.994978. All six early encounters trail the fixed reference. This does not
erase historical scores: it identifies the conditional-prediction acceptance
criterion that was missing when those scores were called progress.

At 35k NLL/reference are 7.789223/7.491462; at 40k, 7.718392/7.230941;
at 70k, 8.380647/7.729342; at 85k, 9.642368/8.301437. A separate older Gemini
unfrozen run also reports plateau NLL 8.204849 at 70k. That separate trajectory
is corroborating chronology, not a matched causal control.

Earlier 3000-step ALIF/STP frozen-window benchmarks reached 7.500371/7.515884.
They use d128, trainable tied token embeddings, four warm-start site windows,
and 128 targets/update (384k training targets). Later online code changed to
d768 with pretrained token embeddings, a separate task head/RMSNorm, local
credit and different optimizer coverage. Those are changes to the learning
contract, not merely larger versions of the old model. Historical 3D-medium
7.10975/7.21536 results belong to a different continuous-medium architecture.

## Earliest mechanism evidence

The 34,190-target audit predates the destination-eligibility upgrade. Motor
centered effective rank is 27.75; read projection yields 2.32 and normalized
latent 1.52, with 99.91% common-direction energy. Writer/edge proposed updates
were largely below float32 resolution while output parameters moved.
Thus source–readout learning imbalance was already present before the later
corrections. Rank alone does not establish semantic usefulness; later mature
full/common readout comparisons directly show negligible conditional output
benefit on the inspected actual revisits.

CPU inspection of the old ALIF/STP best checkpoint confirms its 81 STP
parameters were trained (mean U0=.255385, tau_fac=95.38285, tau_rec=194.3043).
The later online/BPTT lineage excluded them from optimizer updates and retained
.25/100/200 through 1.1M targets. Restoring their trainability repairs a lost
search dimension; its matched 44,832-target prefix changes NLL by only
+0.000073, so it has not demonstrated a prediction remedy on that prefix.

The old output matrix has 165122 columns, but the historical implementation
explicitly masks the read state to output neurons before projection. Its width
does not imply global read access. Current compact 2333-column output readout
preserves that anatomical restriction.

## Decision

The leading measured bottleneck is that motor differences contribute little to
task prediction while a common background dominates output learning. There is
also genuine excess predictive deterioration during sustained online learning.
The onset is known; a single responsible upgrade is not identified by this
unpaired timeline. Destination eligibility and update cadence are candidates,
not established causes.

Next diagnosis should separate effective marginal calibration from the varying
conditional term on matched real continuing histories, and compare the old and
new learning contracts. Preserve sensory-only input, motor-only output, complete
life state and pre-update scoring. A coordinate-centering rewrite alone is an
equivalent function and cannot supply a performance repair. Production training
and its original 30M fresh-token budget are unchanged by this audit.

Independent review: diagnosis_review recomputed the early references and
reviewed chronology, mechanism scope and candidate causes.

Artifacts: `results/published/fly_nll_upgrade_timeline_20261005.json`,
`results/published/fly_stp_joint_matched_prefix_20261005.json`,
`docs/information_boltzmann/FLY_ONLINE_PLATEAU_20261004.md`,
`docs/information_boltzmann/FLY_CONDITIONAL_READOUT_DIAGNOSIS_20261005.md`.
