# Fly connectome -> point-neuron conversion audit (2026-10-12)

Scope: conversion artifacts of MaleCNS v1.0 into `data/malecns_v1/fly_reservoir_coba.npz` (COBA LIF, E_E = 1,
E_I = -0.2, theta ~ 0.1). This audit does not argue the biology. Everything was computed read-only with numpy/pandas, with no torch and no GPU.
The adult's learned weights and physical state were read from `E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt`
by a small zip/pickle reader that never imports torch. The scratch scripts are in `C:/Users/85750/AppData/Local/Temp/claude/flyaudit/` (a2-a16).

Pipeline as built: `build_fly_reservoir_graph.py` (full) -> `build_fly_reservoir_delayed_graph.py` -> `build_fly_reservoir_biological_graph.py`
(DAN split, Scheme-B tau) -> `scripts/ib/build_coba_graph.py` (E/I partition by presynaptic sign).

Severity: **High** = changes what a pathway computes; **Medium** = shifts an operating point or a class-wide balance
that learning or homeostasis can only partly undo; **Low** = cosmetic, provenance, or affects few synapses.

## Summary

| # | Finding | Severity | Minimal fix |
|---|---|---|---|
| 1 | KC->DAN synapses (59% of DAN input, 80% of DAN birth excitation) sit on DAN axon terminals in the lobes but drive the DAN membrane | **High** (dopamine teaching signal) | ROI split as for KC->KC; route lobe synapses out of the somatic drive |
| 2 | No electrical synapses: GF->TTMn is 90 chemical synapses (1.5% of TTMn input), GF->PSI is 16 (0.15%) | **High** (escape pathway) | Add explicit gap-junction edges for GF->TTMn/PSI and the other known couplings |
| 3 | The adult's DN saturation comes mainly from learning. The conversion supplies the predisposition: sqrt(K_in)-tiny per-edge weights plus per-coordinate Adam make input growth scale with K_in^1.4 | **High** for the read surface | Multiplicative (log) edge parameterization or per-edge lr proportional to w0; refractory tick |
| 4 | K_in counts 1-synapse partners: 40% of edges are single synapses, so strong inputs are diluted by a class-dependent 1.6x-5.1x | Medium | Normalize by synapse count, or count partners only at >= 5 synapses |
| 5 | Per-class input totals scale as K_in^0.50: MBON 4.8x, motor neurons 4.6x, DN 3.0x the brain median; sensory 0.13-0.6x | Medium | Per-cell input gain (regime homeostasis `s`) or synapse-count normalization |
| 6 | KC axo-axonal fix (known): confirmed. The renormalization under-corrects by about 8.6% because the birth K_in also counted DAN partners | Low-Medium | Use birth K (full graph) and subtract the DAN partners as well |
| 7 | Sensory terminal input (known): 486k edges, 1.9% of edges; ORN input is 43% ORN->ORN | Low (as found) | Keep the SENSORY_PIN option |
| 8 | APL and CT1 are non-spiking, compartmentalized giants modeled as one spiking point | Medium | Graded output (no spike/reset) or split per compartment |
| 9 | Sign table: unclear NT -> +1 (3,100 cells, 1.65% of output synapses); octopamine/serotonin kept fast-excitatory while dopamine is made modulatory; E/I driving-force ratio is 3:1 at threshold | Low-Medium | Map unclear cells by `celltype_predicted_nt`; treat OA/5-HT like DA, or document |
| 10 | Delays: soma-to-soma Euclidean distance; 15% of somata imputed (all sensory); 83% of edges fall in tier 1 | Low | Acceptable; GF->TTMn needs an electrical (sub-tick) path anyway |
| 11 | DANs have no output at all in the adult: the fast edges were removed, dopamine modulation was not attached, and `dan_plastic_lr` = 0 | Medium (adult only) | Attach modulation, or document that DANs are inert there |
| 12 | Provenance: `build_fly_reservoir_graph.py` cannot have written the stored npz (it uses an undefined `radius`); the stored meta is inconsistent | Low | Fix the script and regenerate the hash |

---

## a) Build pipeline: what was filtered

The meta numbers in the brief (`raw_edges 151,856,684; included_edges 23,229,419; included_synapse_weight 110,466,028`)
come from the **256-parcel builder** (`scripts/ib_local/build_malecns_graph.py`, `malecns_parcels_256*.json`). That builder
keeps only the 140,024 Traced bodies that have a `somaLocation`. **The reservoir uses a different filter.** Measured by
streaming the whole `connectome-weights.feather`:

- Raw: 151,856,684 rows and 311,833,243 synapses. No rows have weight <= 0.
- Nodes: all 165,122 bodies with `status == Traced`. Dropped: Orphan 15,925, Glia 11,864, Unimportant 10,751, None 5,472,
  Assign 1,832, Anchor 611, plus bodies that are not in the annotations at all.
- Kept: 25,563,096 Traced->Traced pairs carrying 124,024,573 synapses (39.8% of raw synapses). In the COBA file,
  25,321,395 of these are fast edges (15,519,257 E and 9,802,138 I) and 241,701 are DAN edges.
- **No weight threshold.** 40.3% of edges are single-synapse, 58.9% have <= 2 synapses, and 75.6% have < 5 synapses
  (these carry 27.7% of the synapses). See finding 4.
- Autapses: only 101 Traced self-pairs (473 synapses, max 30) exist. They are dropped (`pre != post`). Negligible.
- Duplicates: none. Each (pre, post) appears exactly once, so the reduceat aggregation is a no-op.
- Sign per pair: the sign is constant per presynaptic body (Dale's law on `consensus_nt`), so no mixing is possible.
- Missing partners on the input side: 4.9% of all synapses onto Traced cells come from non-Traced bodies (4.3% from
  unannotated fragments). Per superclass: ol_sensory 20.7%, cb_sensory 15.5%, vnc_sensory 15.4%, sensory_descending 19.6%,
  DN 6.8%, unknown 52%, ENS 96%. 58% of the *output* synapses of Traced cells go to non-Traced fragments. That loss is uniform and does not
  enter the K_in normalization.
- Dead nodes: 659 cells have zero input (350 in `unknown`, 139 cb_sensory, 44 ENS) and 1,975 have zero output.
  All 516 `unknown`-superclass cells and all 47 ENS cells have K_in = 0. They are harmless but cost read/write capacity.
- **Provenance (Low):** `build_fly_reservoir_graph.py` writes `"spectral_radius_before_scaling": radius`, and `radius` is
  undefined, so this script cannot have produced the npz on disk. The stored meta says 4.36e7 for the radius, but stored
  g = 0.005761 implies an unscaled radius of 173.6, and the 4.36e7 figure looks like the radius of the raw count matrix. In addition, power iteration
  on a signed, non-normal matrix does not reliably converge to the spectral radius. Only the global scale is affected, and the learned
  `g_e`/`g_i` absorb it, but the graph is not reproducible from the repository as it stands. Fix: correct the variable, rerun, and compare `weight_digest` (f555684bd00dd7d9).

## b) Transmitter sign table

`NT_SIGN`: ACh +1; GABA, glutamate, histamine -1; dopamine, octopamine, serotonin, unclear +1; a missing NT row also gets +1.

| consensus_nt | cells | sign | note |
|---|---|---|---|
| acetylcholine | 103,718 | + | median prediction confidence 0.96 |
| glutamate | 29,296 | - | 19,037 OL, 6,705 CB, 2,103 VNC; includes **302 vnc_motor + 4 cb_motor MNs** |
| gaba | 22,055 | - | |
| histamine | 5,910 | - | 4,114 photoreceptors and 1,777 **T1** (T1 has 4 output synapses per cell, so this is harmless) |
| unclear | 3,100 | + | median cell-type confidence 0.48; 389 vnc_motor, 80 cb_motor, 547 vnc_sensory, 481 VPN; **297 have celltype_predicted_nt = serotonin** |
| MISSING | 502 | + | none of these cells has a single output synapse, so this is moot |
| dopamine | 392 | (removed to DAN channel) | 384 CB and 8 visual_centrifugal |
| octopamine | 101 | + fast | 37% have cell-type confidence < 0.5 |
| serotonin | 48 | + fast | |

There are 107,861 + cells and 57,261 - cells. Ground truth exists for 83,496 cells and never disagrees with the consensus.

- **Photoreceptors (93% inhibitory input) are not a mislabel.** Of the 215,825 synapses onto the 4,107 R cells,
  Dm9 (glutamatergic) supplies 141,282 (65.5%), and R7<->R8 histaminergic synapses supply 22.6%. C2/C3 GABA adds about 2.7%. Both the Dm9
  feedback and the reciprocal R7/R8 histamine inhibition are documented real interactions. R1-R6 differ: only 24% of their input
  is inhibitory, and 76% is ACh (L4, L2, L3 feedback). The actual artifact is the compartment, not the sign. All of this input lands on photoreceptor
  terminals (see section d) and is removed by SENSORY_PIN.
- Inconsistency (Low-Medium): dopamine is treated as modulatory, while octopamine (467,888 output synapses) and serotonin
  (132,786) remain fast excitatory. "Unclear" contributes 2.05M output synapses (1.65%) as excitation. For
  motor neurons the sign hardly matters, because their central outputs total only 68.7k synapses.
- Driving-force asymmetry (Medium, a conversion parameter): at v = theta = 0.1, the excitatory driving force is
  E_E - v = 0.9 and the inhibitory one is v - E_I = 0.3, a ratio of 3:1. Mapping rest = -55 mV, theta = -45 mV, E_ACh ~ 0 mV,
  E_Cl ~ -70 mV gives about 45:25 = 1.8:1. Per unit conductance, inhibition is therefore about 40% weaker than the biophysical
  mapping, and at v = 0 it counts only 0.2x. The learned g_i/g_e = 1.15 does not close that gap.
- Fix: for `unclear`, use `celltype_predicted_nt` where it exists. Either treat OA/5-HT the way DA is treated or document why they stay fast.
  Consider E_E ~ 0.55 at the current theta, or re-derive E_I.

## c) Per-class input totals (birth weights, sum over edges onto each cell)

Brain median: sum|w| = 0.2045 (E 0.110, I 0.094), K_in = 100, 349 synapses in. Across cells, synapses in scale as K^1.007 and
sum|w| scales as **K^0.504**. That is the sqrt(K_in) rule: total drive grows as sqrt(partners) times the mean synapses per partner.

| group | n | median sum|w| / brain | inhibitory fraction | K_in | likely cause |
|---|---|---|---|---|---|
| MBON | 97 | **4.8x** | 0.19 | 937 | very high K (KC fan-in), with strong dendritic input |
| vnc_motor / cb_motor | 708 / 107 | **4.7x / 4.5x** | 0.44 | 250 / 203 | 2,600 synapses in; large dendrites |
| ALON / vnc_efferent | 14 / 94 | 3.8x / 3.5x | 0.29 / 0.40 | 420 / 160 | |
| descending_neuron | 1,314 | **3.0x** | 0.41 | 318 | 2,007 synapses in |
| ALPN / ALLN / ascending | | 2.1-2.2x | | 190-261 | |
| Kenyon_Cell | 4,064 | 0.89x | 0.13 | 190 | before the KC fix; 55% of input is axo-axonal |
| visual (photoreceptors) | 4,107 | 0.63x | **0.97** | 5 | Dm9 + R7/R8 terminal input |
| olfactory / tactile / proprio | | 0.52-0.59x | 0.27-0.81 | 26-57 | 11-20% of input partners are missing (non-Traced) |
| vnc_sensory / mechanosensory / cb_sensory class | | **0.37x / 0.26x / 0.13x** | | 10-23 | few partners; 15-31% missing |
| `unknown` (516), ENS (47) | | 0 | | 0 | no Traced inputs |

Severity Medium. Per-superclass learned theta and tau, together with the global g_e/g_i, can only shift whole classes. A 5x spread
*within* the read surface (MN versus DN) and the high-K cells are what saturate first (finding 3). Fix: give each cell
an input gain (the regime homeostasis `s` already does this in later lineages), or normalize by sqrt(synapses in) instead of
sqrt(partners).

**Weak-partner dilution (finding 4).** If K_in counted only partners with >= 5 synapses, the strong edges onto a cell would gain
sqrt(K/K5). The median gain by class is MN 1.56, MBON 1.59, DN 1.91, ALPN 1.91, ORN 2.89, **KC 3.76, DAN 5.08** (DANs have
482 partners, of which only 19.5 have >= 5 synapses). Single-synapse partners therefore shift relative drive between classes by up to about 3x. Their
contribution to the drive is real but tiny, yet each one enlarges the denominator as much as a 50-synapse partner does.

## d) Compartment artifacts (point model treats every synapse as somatic)

There is no per-synapse ROI table locally, apart from `kc_kc_dendritic_synapses.feather`. As a proxy for location I used the presynaptic
cell's superclass: a brain-only presynaptic cell implies a brain synapse, and a VNC-only one implies a VNC synapse. The table gives input-synapse shares.

| target | from KC | ALPN | ALLN | DAN | APL | DN | AN | brain-only | VNC-only | sensory |
|---|---|---|---|---|---|---|---|---|---|---|
| KC (4,064) | **0.554** | 0.188 | | 0.108 (mod.) | 0.094 | | | 0.050 | | |
| DAN (340) | **0.589** | 0.008 | | 0.020 | 0.004 | | | 0.352 | | |
| APL (2) | 0.899 | 0.023 | | 0.035 | | | | 0.033 | | |
| MBON (97) | 0.656 | | | 0.056 | 0.006 | | | 0.240 | | |
| ALPN (686) | 0.005 | 0.107 | 0.346 | | 0.003 | | | 0.116 | | 0.418 |
| DN (1,314) | | | | | | 0.131 | 0.125 | 0.662 | **0.032** | 0.044 |
| AN (1,846) | | | | | | 0.110 | 0.169 | **0.056** | 0.423 | 0.241 |
| vnc_motor (708) | | | | | | 0.086 | 0.051 | | 0.833 | 0.029 |
| sensory (15,786) | | | 0.136 | | | 0.024 | 0.070 | 0.148 | 0.211 | 0.405 |

- **KC->KC (known artifact 1, verified).** There are 642,932 edges carrying 1,153,844 synapses, 55.4% of all KC input. The calyx (dendritic) share is
  2.94% (33,915 synapses on 30,817 pairs; the feather matches the graph weights 100%). 612,115 edges (95.2%) are pinned at 0.
  The renormalization factor sqrt(K_all/K_dend) has a median of **2.886**. K_all is taken from the fast-only `edge_post` (median 162), but the birth
  normalization divided by the full K (190), which includes a median of **35 DAN partners** that are not fast inputs. The correct factor is
  3.16, so the KC inputs are still about 8.6% too weak. Fix: in `renormalize_kenyon_inputs`, take K_all from the birth graph
  (`fly_reservoir_full.npz`) and subtract the DAN partners along with the axo-axonal KC partners.
- **KC->DAN (new, High).** 281,862 KC synapses (58.9% of DAN input) supply a median **80.5% of each DAN's birth
  excitatory weight**. These synapses lie on DAN axon terminals inside the MB lobe compartments, which makes them axo-axonal. In the point model,
  any KC population activity (that is, the odour itself) depolarizes the DAN soma and makes it spike into *every* compartment it innervates. This mixes
  the CS into the reinforcement channel, which is exactly the path that `attach_compartment_memory`/`online_da` rely on. The same problem
  applies to DAN<-MBON synapses in the lobes. Fix: run the `neuprint_kc_kc_rois.py` query for (KC -> DAN) and (MBON -> DAN) pairs, then pin or
  re-route the lobe synapses, for example into a terminal-local release factor, as was done for KC->KC.
- **APL (Medium).** There are 2 cells, each with 3,031-3,154 partners, receiving 234k input synapses (90% from KCs) and making about 210k output synapses. APL is
  non-spiking and acts locally (each calyx and lobe region is fed back independently). As a spiking point, it is either globally silent or emits
  global, all-or-none inhibition onto every KC (median 50 APL synapses per KC, against 97 PN synapses). Fix: give APL a graded output, i.e. transmit a rectified
  v without spike or reset, or split it into calyx and lobe units using ROI data.
- **CT1 (Medium, vision).** There are 2 cells with about 11.4k partners and 155k input synapses each. In reality it is a non-spiking, column-local amacrine-like cell; as a point
  it becomes a whole-eye integrator. The same fix applies.
- **ALPN axon terminals (Low).** Input from outside the AL is 11.6% brain-only, 0.5% KC and 0.3% APL, spread over many types (the
  largest, LHPV12a1, is 0.5%).
- **MBON (Low).** 82% of birth excitation comes from KCs onto the dendrites, which is correct. Non-KC input is diffuse.
- **DN axons in the VNC (Low-Medium).** The lower bound on axonal input (from VNC-only cells) is 3.2% (per-cell median 1.7%, 90th percentile 7.5%). DN->DN
  (13.1%) and AN->DN (12.5%) are partly in the VNC, giving an upper bound of about 29%. This is not enough to explain the saturation (see section g).
- **AN terminals in the brain (Low).** Brain-only input is 5.6%.
- **Sensory terminals (known artifact 2, verified).** The 15,896 sensory cells receive 331,468 E edges and 154,655 I edges (1.9% of edges, 2.8% of E
  weight, 6.0% of I weight). ORN input is 42.8% ORN->ORN, and the lLN types make up most of the rest. SENSORY_PIN zeroes all of it.
- **Motor neurons (Low).** 83% of their input comes from VNC-only cells onto dendrites, which is correct.
- neuPrint queries needed (no network access was used): per-pair `roiInfo` split, as in `scripts/ib/fly_interface/neuprint_kc_kc_rois.py`, for
  `(:Neuron {class:'Kenyon_Cell'})-[c:ConnectsTo]->(:Neuron)` restricted to DAN and APL targets (split MB lobe/peduncle ROIs from CA),
  APL->KC by ROI, DN inputs by VNC versus brain ROIs, AN inputs by brain ROIs, and ALPN inputs in AL versus CA/LH.
  `MATCH (a)-[c:ConnectsTo]->(b) WHERE a.bodyId IN $pre AND b.bodyId IN $post RETURN a.bodyId, b.bodyId, c.weight, c.roiInfo`.

## e) Delays

Rule: delay = round(0.8 ms + |soma_pre - soma_post| / 300 µm/ms), clipped to [1, 4]. The longest distance is 1,011 µm, so the clip never binds.
The tiers hold 82.7% / 15.1% / 1.8% / 0.4% of edges, with distance bands 0-210 / 210-510 / 510-810 / 810-1,011 µm. 15.2% of somata are imputed (one-hop
mean of partner somata). All 15.9k sensory neurons are imputed (ORNs have 0% known somata), and 8,338 OL cells. 9.3% of edges involve
an imputed coordinate.

| pathway | edges | tier 1/2/3/4 | median distance (µm) |
|---|---|---|---|
| DN -> VNC | 263,711 | 0.00/0.30/0.59/0.12 | 597 |
| AN -> brain | 229,449 | 0.00/0.31/0.49/0.20 | 603 |
| ORN -> PN | 33,821 | 1.00/0/0/0 | 93 |
| PN -> KC | 22,586 | **0.54/0.46**/0/0 | 208 |
| KC -> MBON | 61,210 | 0.74/0.26/0/0 | 178 |
| GF -> TTMn | 2 | 0/1/0/0 | 500 |

The magnitudes are sane for a 1 ms tick. Two artifacts are minor. First, PN->KC synapses all form in the calyx but are split between 1 and 2 ticks by soma
geometry, because soma distance does not measure axon path length. Second, GF->TTMn takes 2 ticks where the real path is a sub-millisecond electrical synapse. Severity Low.

## f) Electrical synapses

There are none: the dataset contains chemical synapses only, and the builders add none. Measured cases:
- **Giant fiber escape.** GF (DNp01, 2 cells, both drawn from body 10001/10010) -> TTMn has 90 chemical synapses on 2 edges, 1.5% of TTMn's 6,032
  input synapses. GF->PSI has 16 synapses, 0.15% of 10,835. In the fly, both connections are dominated by shakB gap junctions. In the model, the
  escape command is functionally almost disconnected from the jump/flight motor neurons. This matters because the repository uses GF as the go neuron
  (`attach_connectome_go`, `GIANT_FIBER_BODY_IDS`). Severity **High** for any escape or "go" reading via TTMn/PSI.
- Other known gap-junction circuits that are missing: lobula-plate HS/VS chains and their coupling to neck-motor DNs (DNOVS), excitatory LN-PN coupling in
  the antennal lobe, clock-neuron coupling, and parts of the GF-giant-commissural and lamina circuits.
- Fix: add an electrical edge list with a symmetric current g_gap (v_pre - v_post) applied in the same tick, starting at least with GF<->TTMn,
  GF<->PSI and GF<->GF. Its conductance would be a parameter, because the connectome gives no number for it.

## g) Other: why ~10% of the adult's DNs fire on every tick

The snapshot comes from the physical state in `last.pt`, using the learned weights, g_e = 0.960, g_i = 1.104, and theta 0.093-0.114 by superclass.
I evaluated the drive one tick after a reset, v = beta_int (G_E - 0.2 G_I) > theta + beta_a b.

- 145 DNs (11.0%) and 60 cb_intrinsic cells (0.19%) meet that condition. **No other class has any such cell.** This matches the
  odour_chain measurement (90th-percentile DN rate 0.99, 82% of DNs below 5%). The exact rates of 0.5 and 1.0 (DNa03, MDN) follow from reset-to-0 with no
  refractory period.
- The excitatory drive on the saturated DNs is **24x its birth value**. Recomputed with birth weights, the same presynaptic activity would
  reach only about 0.1x theta (median). 79% of the drive comes from cb_intrinsic cells, 8% from VPNs and 6% from other saturated DNs.
- Which predictors separate saturated from quiet DNs (AUC): learned E-weight 0.999, learned/birth ratio 0.978, **K_in 0.944**, birth E-weight
  0.832. Median K_in is 930 for saturated DNs versus 255 for quiet ones.
- Mechanism (a conversion x optimizer interaction): the sqrt(K_in) birth rule makes per-edge weights tiny, with a median E edge of 7.4e-4, which is
  comparable to the learning rate of 2e-4. Per-coordinate Adam steps ignore that scale. The 90th-percentile |Δw| is 0.023, against a 90th-percentile birth weight of 0.0037. So a cell's
  learned input growth scales with its edge count: ΔW_in ∝ K_e^1.40 for DNs and ^1.56 brain-wide, whereas birth drive ∝ K^0.6-0.8.
  Read-surface cells receive the read gradient, and their median ΔW_in is 1.56, against 0.05 for non-read cells (30x).
  High-K DNs on the read surface therefore run away. Weak inhibition (E_I, section b) and the absence of a refractory period let them stay at one spike per tick.
- Conclusion: this is not a pure conversion artifact. Axo-axonal VNC input is 3% or less, and the signs are normal. But the conversion's
  normalization is what lets per-edge learning convert in-degree into drive.
- Fix: parameterize learnable edges multiplicatively (w = w0 exp(θ)), or scale the per-edge lr by w0 (equivalently by 1/sqrt(K_in) per target),
  or learn a per-cell input gain instead of free per-edge magnitudes. Add a 1-tick absolute refractory period as a guard. The regime
  homeostasis of later lineages also caps it.

Other items:
- **DANs are output-dead in this adult (Medium).** `build_fly_reservoir_biological_graph.py` moved all 241,701 DAN edges out of
  the fast list, and the COBA step never reads `dan_edge_*` (only the CUBA branch does). This adult's config attaches no dopamine
  modulation and has `dan_plastic_lr = 0`. The 392 DANs (0.47% of synapses) therefore influence nothing in it, even though they still receive
  input and fire.
- Scheme-B tau_m = 3 + 75 K/(K + K_mean) is an in-degree mapping, not biology. In the adult it is used only to initialize the per-superclass learned tau
  and to set homeostasis targets (nu* = duty / tau_m), so it remains a hidden in-degree dependence in the later lineages.
- Learned edge weights stay non-negative (no sign flips; `edge_positive` is kept). 325,600 E edges were learned down to exactly 0.

## Verification notes

All numbers were recomputed from `connectome-weights.feather`, the annotation and NT feathers, `fly_reservoir_coba.npz` /
`fly_reservoir_full.npz`, `kc_kc_dendritic_synapses.feather`, and the checkpoint's `model` and `learner.physical` tensors. Neither GPU nor torch was used.

## Status 2026-10-13: fixes graded by the active route

Principle (user): the fly brain is a prior for an evolving infinite-stream information life, not a replica; fix what
blocks the main mechanisms on the way to activeness, coarse-grained, without slowing compute or raising memory;
biological limits with faster digital equivalents need not be copied.

| Artifact | Blocks | Fix | State |
|---|---|---|---|
| Kenyon axo-axonal KC->KC as somatic; sqrt(K_in) counted axo-axonal and dopamine partners | odours never reach Kenyon cells | `attach_kenyon_dendritic_only` + `renormalize_kenyon_inputs` (once per lineage) | done |
| Kenyon->dopamine terminal synapses as somatic (80% of DAN excitation) | teaching signal follows text, not outcomes (outcome burst indistinguishable from odour-only) | `pin_kenyon_to_dopamine` | done: outcome burst 2.4x, pairing writes odour-specific depression (7x) |
| Absolute Adam steps vs birth weights (median 7.4e-4 vs lr 2e-4) | saturated DNs, antennal lobe grown into a self-exciting loop (E x5-8, I x1.3-2) | relative plasticity (`--relative-edge-rate 0.02`); antennal-lobe inputs back to birth once (`--al-reset`) | done: PN baseline 0.12 -> 0.017 |
| Compartment rule: signed fast-slow phasic (integrates to ~0), forgetting tied to the rate | memory not written / not kept | rectified phasic, depression x (1 + phi), slow forgetting | done |
| Synapses onto sensory terminals as somatic | receptor self-excitation | pinned (`pin_sensory_terminals`); release modulation later if needed | done (coarse) |
| Octopamine / serotonin as fast excitation (OA-VPM3/4, VUMa2 drive Kenyon cells) | Kenyon background | fast edges pinned (`pin_modulatory_fast_edges`, neuPrint consensus NT); modulation not modelled yet | done (coarse) |
| APL as a spiking point (29-45% firing, STP depletes it) | Kenyon sparseness / specificity (d' 0.32) | graded APL | conditional: only if evolution does not sharpen the Kenyon code |
| No gap junctions; global sqrt(K_in) single-synapse dilution; weak inhibitory driving force; unclear-NT cells excitatory; soma-distance delays | not on the active route now | none | deferred |

Cost: pins and resets zero weights only -- same speed and memory (smoke run: 3.1 GB peak).
