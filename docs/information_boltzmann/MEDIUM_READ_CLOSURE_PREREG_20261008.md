# Local read closure versus additional temporal representation

Registered before execution; user authorizes distinguishing the two candidates
without training. CPU native, complete last individual, no parameter updates.
No new read head is fitted, no SVD/regression learner or optimization is used.

Hypotheses:

- Existing instantaneous field omits dynamical variables needed to describe
  future local responses. Exposing existing motion/state may repair this.
- Even all state inside a finite aperture is only a partial observation: signals
  crossing its boundary can affect later responses. A spatial boundary halo or
  temporal observer are alternative ways of dealing with that limitation.
- Additional temporal statistics may encode discarded external-input history.
  This is a new memory/representation hypothesis, not missing physical initial
  conditions. It cannot be accepted/rejected from physical forecast accuracy.

Mathematical scope: for fixed weights, solver and specified future controls,
complete physical state determines future physical state. This does not claim
sufficiency for external next-token prediction or the future of online learning:
the latter also needs the actual optimizer/pending-credit state. No 'CTM is
redundant' or 'full-state read wins language' verdict is permitted.

Use the next32 actual OWT events, preserving the main .005 trajectory. At
preselected events0/8/16/24, branch after the observed carry/current token has
been written. Do not feed any future token into a branch.

1. Capture the existing16 local probe measurements (mean/variance, before
   their merge). For two deterministic random scalar projections per probe,
   compute the independent complete-state gradient of the same measurement
   after one .005 evolution. Split squared gradient support inside the aperture,
   inside one periodic nearest-neighbor halo, and outside. Treat edge states
   as visible when either endpoint touches the site region; compare fractions
   within each component, not dimensionally incompatible norms across groups.
   These are sampled VJPs, not an exhaustive output Jacobian certificate.
2. Pick the smallest-support probe by geometry, with no label access. For that
   probe, perturb only flux edges outside the aperture/halo by +/-1%, keeping
   its entire candidate local state unchanged. Verify instantaneous measurement
   equality and report future differences. This is a legal numerical state
   sensitivity test, not a synthetic task or trained capability evaluation.
3. Compute autonomous df/dtau at0 from the existing complete state using exact
   JVP. Compare the existing local read applied to f+epsilon*df/dtau against
   the actual read after epsilon=.005/4,.005/8,.005/16. Current f-only hold is
   the reference. Error ratios when halving epsilon distinguish a usable
   first-order dynamical observable from an arbitrary extra feature vector.
   Compare normalized feature and local measurement errors separately.

Decisions: first-order correction with approximately quadratic residual favors
testing a local motion-aware interface before claiming new neural history is
necessary for response prediction. Outside-aperture sensitivity requires a
boundary closure condition; history is one possible observer, not automatic
truth. Inconsistent JVP/error scaling triggers numerical audit, not training.
External language gains and history's additional conditional information remain
unknown under the user's no-training constraint.

Independent plan reviewer /root/rtc_contract_review accepts these first two
closure/derivative tests, emphasizing boundary-edge classification, quiet
trajectory derivatives, fixed-parameter scope and no linguistic victory claims.
