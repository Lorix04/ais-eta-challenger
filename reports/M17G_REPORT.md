# M17G — Stronger Self-Supervised AIS Encoder: MoCo / Masked Autoencoder vs frozen M17A

## Decision

**PASS_STRONGER_SSL_ENCODER_COMPONENT**

Frozen M17A MAE: **190.105 h**.

- moco_tcn64: **188.287 h MAE**, gain 1.819 h, fold wins 3/5, P90 ratio 0.9533, gate=True.
- masked_transformer_mae64: **190.910 h MAE**, gain -0.805 h, fold wins 2/5, P90 ratio 1.0054, gate=False.

M17G changes only the target-free trajectory encoder/similarity representation. The M17A destination gate, K=9 retrieval and weighted-median ETA aggregation remain fixed. Each encoder is retrained separately inside each outer fold using only causal windows owned by outer-train MMSIs.

Research references: MoCo-AIS (arXiv:2606.17978); NaviSight (github.com/Korunil/navisight); Representation Learning for Maritime Vessel Behaviour (JMSE 2026, 14(5), 507).
