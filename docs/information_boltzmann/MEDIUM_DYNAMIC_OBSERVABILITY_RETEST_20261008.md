# Retest the diagnosed read blindness without training

The previous `audit_medium_time_read_credit.py::read_observability` held the
current field fixed and changed persisted flux by +/-1%. Old read features
remained identical while future output changed. This is the structural blind
direction to retest, not a language-capability training claim.

Three arms share the same completed checkpoint, all original weights, physical
state, geometry and evolution: old instantaneous read; zero-projection migrated
dynamic read; and migrated dynamic read with ONLY the three new projections
reset to ordinary fan-in initialization with fixed seed 711. The last arm tests
available information flow without optimizing it. Its task quality is unknown.

Advance the same next 32 actual OWT inputs without parameter updates. Select
events 0,8,16,24 in advance; each uses carry-correct current-input/next-target
alignment. Run the original observability function for each arm. Also inspect
new motion means/variances before the added projection for all 16 probes.

Predictions:

1. Old and zero-migrated immediate final-feature flux sensitivity stay zero.
   Their outputs should be identical; function-preserving migration is a control.
2. Active dynamic immediate features and CE partials to persisted flux become
   nonzero. Raw motion measurements should respond even in zero migration.
   The size of response is measured; sensitivity alone is not task benefit.
3. A selected probe remains blind to changes outside its local/incident-edge
   support. This rejects an unintended global read shortcut.
4. The existing first-order physical response check should reproduce: dynamics
   was not changed. Its earlier accuracy gain was the field-motion oracle's
   benefit, not an error score that any new reader must automatically reduce.

If 2 fails, repair the information path before any training. If 1 or 3 fails,
repair compatibility/locality. If these pass, the blind direction is removed
from the explicit active branch's search space; learned usage and NLL benefit
remain a joint-training decision. Zero optimization, no test-label tuning.

## Results

CPU retest completes in approximately 7.4s, 32 actual OWT inputs, 4 preselected
events and zero optimizer updates. All original weights agree across arms;
selected physical transitions are exactly equal for field, flux, conduction,
receptors, STP, elapsed time and precision. Every arm's parameters remain
unchanged after initialization. The original diagnostic is reused directly.

| Read arm | Mean immediate max feature change, flux +/-1% | Mean immediate task partial-gradient norm, flux axis |
| --- | ---: | ---: |
| Old field-only | 0 | 0 (unused) |
| Zero-migrated dynamic | 0 | 0 (connected, zero projection) |
| Dynamic, standard initialization of three new projections | 0.002122 | 1.099584 |

The active arm's final feature relative change is 0.1225%–0.1608% across the
eight predeclared perturbations. Both dynamic arms expose changed raw motion
measurements at all 16 probes for every perturbation. Exact zero output in the
zero-migrated arm confirms compatibility and shows that its added projections
must learn before new information affects predictions.

For the geometry-selected smallest probe (12 sites), nonempty flux perturbation
outside its local/incident-edge support produces exactly zero raw probe change
in both dynamic arms. The previous first-order physical-response errors are
reproduced exactly (maximum feature-jet error difference to the historical
report = 0). Existing physics already provided the response; the repaired active
read path now observes it and receives an immediate task gradient.

These are sampled structural-response and learning-path results. The active
projection initialization is untrained; task benefit remains a sufficient-budget
joint-training question. This repairs the tested flux blind direction; the
derivative is a state summary, not a proof of observing all hidden information.
Gradient norm and feature change use different units and are not signal-to-noise
or usefulness scores. Independent result review by `/root/rtc_contract_review`
accepts these conclusions and requested the complete-state equality check above.
