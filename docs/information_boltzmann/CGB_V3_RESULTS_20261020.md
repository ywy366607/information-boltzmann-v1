# CGB V3 - the medium's rules against the fly's wiring (2026-10-20)

V2 grew, from conserved pools alone, a selection rule: trophic support is captured by plasticity tags, the sliding
threshold makes the tags correlation-specific, so connections whose partners fire together are kept and grow
(the classical name is Hebb; the symmetric, phase-only version is adaptive Kuramoto coupling). V1 left ~52 points of
reproducible connection strength that geometry (>= 2 um), the type code and local crowding do not explain. V3 asks
whether the V2 rule accounts for part of that residual.

## V3a - shared input as a proxy for correlated activity (no simulation)

The dominant source of correlated firing between two cells is common input. If strength grows with correlation,
a -> b should be stronger than geometry predicts when a and b share upstream partners.

Footing: E1d (right-side log synapses per cell pair, right-connected held-out pairs, E1 split, degree-only null,
share of squared error explained on the test pairs). Baseline: directed geometry at 4 + 2 um (E1d: 0.316 test;
replicate ceiling 0.836).

Features, all from the LEFT hemisphere (the right target never enters them), the pair itself excluded:
- SI_ab shared input: cosine of the Hellinger input profiles of a and b (sqrt of the share of each type's input coming
  from each upstream type), upstream types a and b removed from both profiles
- SO_ab shared output (specificity control): the same on output profiles, downstream a and b removed
- SI_perm: SI with types relabelled by a random permutation (same values, wrong pairs)
Each enters as [x, log x (floored), x == 0].

Predictions written before running:
- P1 SI adds >= 3 points of test strength beyond geometry 4 + 2 um
- P2 SI adds more than SO (correlation comes from shared input; shared output does not correlate the two partners)
- P3 SI_perm adds less than 0.5 points (the gain is not a fitting artefact)
- P4 symmetric or causal (Kuramoto vs STDP), no direction predicted: on reciprocal pairs connected both ways on both
  sides, correlate the right-side geometry residual of a -> b with the left-side residual of b -> a (cross-side, so
  shared counting noise cannot correlate them). Reading: clearly > 0 = the strengthening is symmetric in the pair
  (rate correlation, Kuramoto-like); clearly < 0 = one direction wins (causal timing, STDP competition); ~0 = neither.

Limits stated before running: shared input is a static proxy; a positive P1/P2 is a necessary condition for the V2
rule, not proof (a developmental rule that wires shared-input partners together would give the same motif). A negative
P1 removes activity correlation as the source of the residual and leaves sub-2-um contact and per-contact synapse number.

### V3a results (script `scripts/ib/generative/v3a_shared_input.py`, `results/published/v3a_shared_input_20261020.json`, 5 min)

Test strength explained (E1d footing; geometry 4 + 2 um = 0.316, replicate ceiling 0.836, gap 0.52):

| model | test | gain over geometry |
|---|---|---|
| geometry + SI (shared input) | 0.329 | +1.3 pt |
| geometry + SO (shared output) | 0.337 | +2.1 pt |
| geometry + SI_perm | 0.317 | +0.05 pt |
| geometry + SI + SO | 0.345 | +2.9 pt |
| geometry + replicate + SI | 0.839 | +0.01 pt over geometry + replicate |

- P1 **failed**: shared input adds 1.3 points, not >= 3; it closes 2.5% of the 52-point gap.
- P2 **failed**: shared output adds more than shared input (2.1 vs 1.3); SI and SO correlate 0.50 on connected pairs.
- P3 passed: the permuted feature adds nothing (the small gains are real, not fitting noise).
- P4 (47 441 reciprocal type pairs connected both ways on both sides): reverse-direction cross-side residual
  r = **+0.33** [0.33, 0.34], same-direction r = 0.91, shuffled 0.007. With r_ab = s + d, r_ba = s - d this gives
  (var s - var d) / (var s + var d) = 0.33 / 0.91 = 0.36: about **two thirds of the reproducible residual is shared by
  the two directions of a pair, one third is direction-specific**. The within-right decomposition agrees
  (var s 0.95, var d 0.47 -> 67% symmetric). Not causal-STDP competition (that would give r < 0).

Exploratory, post hoc (scratchpad `v3a_sym.py`, `F:/fly_stream_overflow/cgb_v1/v3a_symmetric_exploratory.json`;
not a confirmed result): the symmetric part s does **not** rise with shared input (Spearman -0.07); it rises weakly with
shared output (+0.13). Over all connected pairs the geometry residual vs SI is -0.05, vs SO +0.08.

Reading:
1. Correlated activity from common input - the textbook Hebbian source, and what the V2 rule would amplify - explains
   at most 1-3 of the 52 missing points at the type level. The V2 selection rule is not the main source of the
   residual. Caveat: type-level shared input is a coarse proxy (it ignores sign, timing and rates).
2. The residual is a reproducible, mostly **symmetric pair affinity**: whatever makes a -> b stronger than geometry
   predicts also makes b -> a stronger. A symmetric pair factor fits two candidates better than activity: contact
   closer than 2 um (two neurites that touch touch each other) and molecular pair matching that is full-rank rather
   than low-rank (the k = 16 code adds only 5 points; homophilic adhesion is symmetric by construction).
3. Kuramoto-like symmetric coupling is the shape of the residual, but its driver is not common input.

Next exits: (a) a symmetric sub-2-um contact proxy (undirected cable overlap at the finest scale the skeletons allow)
against the symmetric part s - predicted to explain s if contact is the cause; (b) a full-rank symmetric affinity
model (s fitted as a symmetric matrix with held-out pairs) to measure how much of s is pair-specific rather than
type-code-like.

## V3b - is the symmetric pair affinity contact? (exit (a), pre-registered 2026-10-20)

Derivation (stereology, Buffon / Peters): two randomly oriented neurites, cylinders of radius r_a and r_b, touch where
their axes pass within r_a + r_b. In a voxel of volume V the expected number of touches between the cable of a and of b
is proportional to L_a L_b (r_a + r_b) / V. Summed over voxels:
  contact K_ab = sum_v [L_a(v) LR_b(v) + LR_a(v) L_b(v)],   LR = sum of (segment length x radius) in the voxel
  undirected cable overlap U_ab = sum_v L_a(v) L_b(v)
Both are symmetric in a and b by construction. Skeleton segments are cut into pieces <= 0.5 um before voxelising
(median node spacing 0.72 um); scales 2 um and 1 um; radii from the skeletons (8 nm units, coarsely quantised: the
median equals the 10th percentile, so the radius term carries little information - Pa4 is the weakest prediction).

Footing: E1d (right-side strength, held-out right-connected pairs), baseline = directed geometry 4 + 2 um (0.316).
Predictions written before running:
- Pa1 geometry + K at 1 um adds >= 3 points of test strength
- Pa2 the gain from K grows from 2 um to 1 um (closer to true touching)
- Pa3 contact absorbs the symmetric pair affinity: the reverse-direction cross-side residual correlation of V3a P4
  (+0.33 after geometry) falls to <= 0.22 after geometry + K 1 um, while the same-direction correlation stays high
- Pa4 K (with radius) beats U (length only) at the same scale
If Pa1 and Pa3 fail, contact closer than 2 um at the type level is not the affinity, and exit (b) (pair-specific vs
code-like matching) becomes the main candidate.
Sampling: whole-brain 1 um has ~2e8 type-voxel entries; the run samples the volume systematically (2 um z slabs with
index mod 8 = 0 and = 4, two independent 1/8 samples). Their agreement is reported as the sampling error; the features
use their sum. Script `scripts/ib/generative/v3b_contact_overlap.py` (checked against a sparse brute force on one shard,
relative error < 1e-6).

### V3b results (`results/published/v3b_contact_20261020.json`, 6 min overlap + 7 min analysis)

Sampling is not the problem: the two independent 1/8 volume samples agree at log r = 0.93-0.95 on connected pairs, and
only 0.4% (2 um) / 1.2% (1 um) of connected pairs have zero sampled overlap.

| model | test | gain over geometry |
|---|---|---|
| geometry + U 2 um | 0.318 | +0.2 pt |
| geometry + K 2 um | 0.316 | +0.0 pt |
| geometry + U 1 um | 0.318 | +0.2 pt |
| geometry + K 1 um | 0.322 | +0.6 pt |
| geometry + K 1 um + K 2 um | 0.322 | +0.6 pt |
| geometry + K 1 um + U 1 um | 0.348 | **+3.2 pt** (weights K +1.46, U -1.43) |

- Pa1 **failed** (+0.6, not >= 3). Pa2 passed only formally (0.6 vs 0.0). Pa4 **failed** at 2 um.
- Pa3 **failed**: the reverse-direction cross-side residual stays at 0.329 after contact (0.335 before; same-direction
  0.912). Contact at 1 um, symmetric by construction, does not absorb the symmetric pair affinity.
- Reading: at the type level, how much two arbors come within ~1 um of each other carries almost nothing that the
  directed 2-4 um overlap has not already given. The symmetric affinity is not a closer-contact effect at this
  resolution (the true touching scale, < 0.5 um, is below what the skeletons give).

Post hoc (not pre-registered): K 1 um + U 1 um together gain 3.2 points with opposite, equal weights, i.e. the feature
is log(K / U) = log of the mean (r_a + r_b) of the cable where the two types overlap: thicker neurites at the overlap,
more synapses per pair (more membrane surface per unit length). Re-predictions written before re-testing it:
- R1 the same at 2 um: geometry + K 2 um + U 2 um gains >= 2 points, with weight on K > 0 and on U < 0
- R2 it does not explain the symmetric affinity either: after geometry + K 1 um + U 1 um the reverse-direction
  cross-side residual correlation stays >= 0.30

Re-test (scratchpad `v3b_radius_recheck.py`, `F:/fly_stream_overflow/cgb_v1/v3b_radius_recheck.json`):
- R1 **confirmed**: at 2 um, geometry + K + U gains +2.1 points, weights log K +0.98, log U -1.20.
- R2 **narrowly failed**: the reverse-direction correlation falls from 0.335 to 0.298, so thickness takes about a
  tenth of the symmetric affinity, not none.
- Scale: the mean r_a + r_b at the overlap of connected pairs is 0.59 / 0.68 / 0.83 um (10th / 50th / 90th pct).
Reading: neurite thickness where two types meet is a real, replicated strength factor (~2-3 points, coefficient
~1-1.5 on log mean radius: more membrane per unit length, more synapse room - a surface-area term the encounter
derivation had left out, since it used cable length only). It is not the symmetric affinity.

## V3c - pair-specific or code-like? (exit (b), pre-registered 2026-10-20)

Question: is the symmetric pair affinity of V3a P4 a low-dimensional matching code (a handful of recognition
molecules shared by many types), or specific to each pair?
Operational test on the E1d footing (baseline directed geometry 4 + 2 um, 0.316):
- reverse: geometry + the right-side strength of the same pair in the other direction, b -> a (log per cell pair and
  a connected flag): the pair-specific symmetric information, whatever its cause
- symmetric code k: geometry + C A C^T with A symmetric, k = 16, 64, 256 (fitted on training pairs, which include most
  reverse directions; l2 chosen on validation from 1e-4 to 10, so a useless code shrinks to zero)
- general code k = 64 (A unconstrained) for comparison; symmetric code 64 + reverse
Expected size from P4: the reverse direction correlates ~0.33 with the residual on reciprocal pairs.
Predictions written before running:
- Pb1 reverse adds >= 3 points of test strength beyond geometry
- Pb2 the affinity is pair-specific: the best symmetric code (k <= 256) recovers less than one third of the reverse gain
- Pb3 the reverse gain survives the code: geometry + symmetric code 64 + reverse keeps >= 2/3 of the reverse gain over
  geometry + symmetric code 64
If Pb2 fails (a small symmetric code captures most of it), the affinity is code-like: few molecules, many types.
Script `scripts/ib/generative/v3c_pair_affinity.py`.

### V3c results (`results/published/v3c_pair_affinity_20261020.json`; l2 grid extended in
`scripts/ib/generative/v3c_code_l2_extend.py` -> `results/published/v3c_code_l2_extend_20261020.json`, because the first
grid chose l2 = 10, its edge)

Test pairs 14 791; the reverse direction is connected for 43%.

| model (best l2 on validation) | test | gain over geometry |
|---|---|---|
| geometry + reverse | 0.342 | +2.6 pt |
| geometry + symmetric code 16 / 64 / 256 | 0.341 / 0.371 / 0.390 | +2.5 / +5.5 / +7.4 pt |
| geometry + general code 16 / 64 / 256 | 0.316 / **0.422** / 0.406 | +0.0 / **+10.6** / +9.0 pt |
| general code 64 alone | 0.217 | (E1d's code k = 16 alone: 0.051) |
| geometry + general code 64 + reverse | 0.434 | +11.8 pt (reverse adds 1.2 beyond the code) |

- Pb1 **failed** narrowly: the reverse direction adds 2.6 points (not >= 3).
- Pb2 **failed**: a symmetric code recovers more than the reverse direction itself (5.5-7.4 vs 2.6). The symmetric
  affinity is code-like (shared by many pairs through the types' identities), not specific to each pair.
- Pb3 **failed**: beyond the code, the reverse direction keeps 1.2 of its 2.6 points (45%): about half of the pair
  affinity is code, the rest pair-specific (or a code we fit badly).
- A directed code beats a symmetric one (+10.6 vs +5.5 at k = 64): matching is mostly heterophilic (a ligand on a, a
  receptor on b - like Dpr/DIP), with a smaller homophilic share.
- Fits are fragile: the best l2 jumps between neighbouring grid values (k = 64: l2 10 gives 0.422, l2 30 gives 0.339),
  k = 16 collapses to zero code for l2 >= 10 (but reached 0.341 at l2 1). The code numbers are lower bounds, not pinned
  values; a careful fit (more steps, seeds, finer grid) would be needed to pin them.
- **Correction to V1 (E1d)**: "code + geometry 0.260 < geometry 0.316" was under-regularisation (l2 <= 0.1). With
  enough shrinkage the type code adds about 10 points of strength beyond geometry. Codes do explain strength - more
  than any single other factor tested after geometry.

Exploratory, all factors together (scratchpad `v3_all.py`, `F:/fly_stream_overflow/cgb_v1/v3_all_exploratory.json`):
geometry + thickness (K, U 1 um) + shared input/output + reverse = 0.384 (+6.8); with a general code 64 (l2 10) = 0.463
(+14.7; train 0.71, so the code still overfits). Against the replicate ceiling 0.836 this leaves ~37 of the 52 points.

## V3 so far - what the strength residual is and is not

Of the ~52 reproducible points geometry left, about 15 are now accounted for:
- type matching code, mostly directed: ~10 (lower bound)
- neurite thickness at the overlap (surface area): ~2-3
- shared partners (input and output): ~3 (overlapping with the code)
- pair-specific symmetric remainder: ~1
Ruled out as main sources: closer contact down to 1 um, common-input activity correlation (the V2 Hebbian rule),
local crowding (E1e). The rest (~37 points) remains open. Remaining candidates: contact below 0.5 um (needs the
segmentation, not skeletons), per-cell variation averaged away at the type level, and a better-fitted code.

## V3d follow-up, completed 2026-10-21

The matching gain is now stable on the unchanged E1 historical split: directed code 64 + geometry explains
42.18% +/- 0.13 percentage points across three seeds, against geometry 31.56%. The gain is 10.62 points.
Symmetric code 64 reaches 38.43% +/- 0.04; directed code 256, with its refined stronger regularization,
42.74% +/- 0.12. Extra updates and cold starts are recorded; all selected grids are interior and the
extra-update objective-change checks pass. Select the 64-dimensional directed rule for the first deployment:
256 gains only 0.56 points for roughly four times the parameters. CPU generators reconstruct all 295982
observed pairs with max error below 6.3e-6. Details and preregistration: `CGB_V3D_RESULTS_20261021.md`.
This follow-up stabilizes the geometry-plus-code model; the earlier 0.463 all-factor combination remains its
separate exploratory result.
