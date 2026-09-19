# Torus3D relative-coordinate ablation at 3000 updates

Both arms use OWT GPT-2 BPE, an 8x8x4 periodic grid, D3Q8 x 16 content
channels, 128 tokens per update, and 3000 optimizer updates.

| metric | absolute-coordinate v2 | relative/coordinate-free v3 | change |
|---|---:|---:|---:|
| best validation NLL | 7.24059 (step 2500) | 7.39198 (step 3000) | +0.15139 |
| final validation NLL | 7.24501 | 7.39198 | +0.14697 |
| four-site causal full NLL | 7.20357 | 7.40233 | +0.19876 |
| collision removal delta NLL | +1.28442 | +5.65153 | +4.36711 |
| transport removal delta NLL | +0.12305 | +0.01812 | -0.10493 |
| joint removal delta NLL | +2.93870 | +7.83434 | +4.89565 |

At the final training update, v3 has higher field energy (0.606 versus
0.214), larger bath outflow (0.00689 versus 0.00142), and a much larger
accepted write fraction (19.6%; the historical v2 log predates that derived
diagnostic). Readout entropy is similar at the endpoint (3.94 versus 4.02),
so readout sharpness alone does not explain the regression.

The v3 change is a bundle of four coordinate treatments:

1. Write centers are displacements from the instantaneous circular energy
   barycenter. This preserves translation equivariance, but the anchor is
   poorly conditioned when energy is diffuse or nearly symmetric.
2. Collision is coordinate-free and becomes the dominant learned operator.
   Its large removal effect shows genuine use, but also compensation for the
   lost spatial route.
3. Bath conductance is coordinate-free and energy-conditioned. It permits a
   substantially hotter operating point and stronger outflow.
4. Readout coordinates are measured from the same instantaneous field
   barycenter. A bulk translation moves both field and readout frame, making
   transport nearly invisible. This is the clearest mechanistic failure.

Conclusion: the bundled relative-coordinate v3 has no performance advantage.
It trades transport-mediated computation for collision-mediated computation
and loses about 0.15 validation nats. Preserve coordinate-free local collision
and bath as symmetry-respecting defaults, but do not center the readout on the
instantaneous energy barycenter. The next controlled arm should change only
the write address while keeping the v2 readout, collision, and bath fixed; a
separate arm can test a persistent transported reference phase for readout.
