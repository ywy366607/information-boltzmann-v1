# 3D medium expression scale

The current plastic 3D medium expresses a measured feature through
`readout -> read_norm -> decoder`. `read_norm` is a final learned RMSNorm;
internal QK normalization and the correction network's internal RMSNorm remain
in their existing roles. All native, timestamped, quiet chunk, online local
credit, UORO, training capture and evaluation capture paths call `model.decode`.

This normalizes the expression interface, not the stored field. Field, flux,
receptor, STP, conduction and precision state retain their physical amplitudes.
The decoder keeps its bias and parameter keys; token-frequency priors are a
separate question from feature-scale calibration.

For a measured feature z, the head uses

    logits = W [g * z / sqrt(mean(z^2) + eps)] + b

with PyTorch RMSNorm's standard all-ones learned gain and dtype epsilon. Positive
rescaling of z is approximately removed away from epsilon. Gain g and decoder W
can still change logits scale, so this is not a proof of bounded logits or of
improved long-term predictive quality. Autodiff provides the norm VJP, including
the learned gain; the CUDA graph captures its ordinary forward and backward.
Reference: Zhang & Sennrich, RMSNorm (NeurIPS 2019), arXiv:1910.07467;
https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html.

New optimizers decay affine/embedding prediction matrices at the existing
configurable rate (default 0.01). Norm gains, biases, attention scales, learned
positions, material coefficients, physical speed/conduction/COBA/STP/bath maps,
and precision maps use zero weight decay. Collision and prediction networks'
matrix weights remain regularized. A log physical rate of zero denotes a
reference rate; shrinking that coordinate is not generic physical shrinkage.
Groups contain parameter names and are recorded in run configuration.

Exact `--resume` preserves the saved architecture, optimizer moments, decay
policy, eligibility, pending gradients and physical belief. A legacy checkpoint
without the constructor flag uses `pre_decoder_norm=False` and its original
one-group optimizer. `--initialize-from` creates an explicit new learning branch
with the norm enabled, existing weights/belief, all-ones new norm gain, and new
optimizer/eligibility. Its data cursor and physical cadence are preserved. Only
the missing final norm weight is permitted during this branch conversion.

Fourth-pillar live health adds raw/normalized feature RMS, vocabulary-logit
standard deviation and predictive entropy in nats from the already-computed
pre-target logits. These are interface measurements; they do not supply an
auxiliary loss or controller. Existing energy/structure histories remain intact.
Sufficient real-data joint training is required for any performance claim.
