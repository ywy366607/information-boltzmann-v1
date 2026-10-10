# Review of the token-conditioned LeJEPA candidate

Status: checked against current implementation and stored run artifacts; conclusion not independently reviewed. No new production experiment launched.

## Observed result

Run results/q8_fly_bptt32_lejepa_scratch_100k completed100000 fresh train targets,3125 train optimizer updates plus240 active-evaluation updates,107680 total physical ticks. First-pass cumulative NLL6.3568805857 versus fixed training-corpus unigram7.4768117810, a paired mean gain1.1199311953. Most recent4096 fresh targets:6.3988856304 versus7.4585546375. Best freshB window6.0163668267 occurs at55008 train targets; final freshB is6.8821936122. These are different measurement objects. The result supports retaining this candidate as a genuine predictive-performance asset.

## Architectural distinction

fly_bptt_learning.py forward_window supplies m.embedding(input_ids) directly to the latent_predictor after motor readout. Since predict(z,e)=z+P(z,e) and blend(z,pred)=z+alpha*P(z,e), the decoder has a token-conditioned route whose current input need not physically traverse sensory-to-motor axons. Conditioning on the observed input for its following target is causal, but differs from the user's motor-only physical routing requirement. Holding motor z constant still leaves a nonlinear input-embedding-to-logits map; the topology alone therefore does not certify the source of the measured gain. This algebra establishes an available route, not its trained share of the gain.

The candidate uses GPT-2 pretrained input AND decoder embeddings, an offline unigram bias, and read_norm initialization0.1. Physical state is fresh; the interface knowledge is pretrained. The fixed reference actually names data/ib_owt_gpt2/train.npy (10940858 tokens), not the full31M corpus. These choices must be matched before attributing improvements solely to JEPA or timing.

## Credit and objective

The implemented auxiliary target is preds[:-1] against normed motor features[1:] within a32-token window. This supplies additional readout-level one-step learning signals; it is not a local error at every node along the25M-edge trajectory. The same BPTT/surrogate mechanism still propagates its gradients. The combined objective is CE+0.1*MSE+0.02*SIGReg (outer lambda_jepa0.1 multiplies internal lambda_sigreg0.2). It is not CE+0.1*MSE+0.2*SIGReg.

A physical response peak13--24 ticks is compatible with a continuing model using past input as later context. Response norm alone does not measure mutual information or determine credit accuracy. Failure to beat unigram after100 updates leaves multiple competing explanations; the existing3000-joint-update minimum remains the capability standard.

## Claims needing revision

A growing mixture alpha supports optimizer preference for the composite predictor; it does not quantify phase correction, information gain or long-path credit. Actual freshB/A2 logs contradict the assertion that all20A2 scores are lower than freshB: at5024,45024 and60000 the A2 mean is higher. A-versus-B comparisons additionally compare different text; savings should use matched A1/A2 targets and actual exposure intervals.

## Next decision

Retain the completed hybrid. Keep the motor-only begin/commit pipeline as the separate causal interface candidate. Match initialization and baseline definitions before claiming a mechanism advantage. The pipeline core passed independent implementation review, but the complete training entry-point approval is pending because the reviewer agents exhausted their account quota. Keep training unstarted until that requested independent gate can actually be completed.

Primary reference on LeJEPA's prediction/anti-collapse objective:
https://arxiv.org/abs/2511.08544
