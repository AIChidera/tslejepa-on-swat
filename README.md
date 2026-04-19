# TS-LeJEPA: Latent-Euclidean JEPA for Industrial Anomaly Detection

**Self-supervised anomaly detection for industrial control systems via latent-space predictive coding.**

> TS-LeJEPA adapts the LeJEPA training objective (Balestriero & LeCun, 2025) to multivariate industrial time series, replacing sensor-space reconstruction with latent prediction. The result: a detection threshold 21.3× tighter than a reconstruction baseline on the SWaT benchmark, with VUS-PR 0.9601 at 1.24M parameters.

---

## Why this work

Reconstruction-based anomaly detectors (USAD, OmniAnomaly, TranAD) all share the same structural problem: to operate stably on real industrial sensors, their detection threshold must absorb the full variance of measurement noise; mechanical vibration, valve actuation, quantisation jitter. That tolerance creates a security blind spot. An attacker who keeps their manipulation below that noise floor is invisible.

TS-LeJEPA sidesteps this by never touching sensor space during inference. The encoder learns to represent the smooth physical trajectory of the process. Inter-stage flow balances, tank level dynamics, valve sequences, and discards noise because noise carries no gradient signal under the JEPA prediction objective. Normal operation becomes tightly predictable in latent space. Attacks break that predictability.

The SWaT results make this concrete:

| Model | Detection Threshold τ | VUS-PR | AUPRC |
|---|---|---|---|
| **TS-LeJEPA** | **0.0622** | **0.9601** | **0.9623** |
| F-USAD | 1.3263 | 0.8831 | 0.8839 |
| F-LSTM | 0.9754 | 0.8570 | 0.8588 |

The gap between τ = 0.0622 and τ = 1.3263 is not a tuning choice — it is structural. F-USAD can never detect an attack whose sensor deviation stays below 1.3263. TS-LeJEPA catches it above 0.0622.

---

## Architecture

```
Raw SWaT window (420 timesteps × 44 channels)
    │
    ▼ 5× downsampling → per-channel RevIN normalization
    │
    ├──── Context segment (360 steps) ──→ LeanEncoder (4-layer Transformer, causal) ──→ z_ctx
    │                                                                                       │
    └──── Target segment (60 steps)  ──→ LeanEncoder (shared weights, causal) ──→ z_tgt  │
                                                                                           │
                                                          TemporalPredictor (1D-CNN) ◄─────┘
                                                                    │
                                                                    ▼ z̃_tgt
                                                                    │
    Loss = MSE(z̃_tgt, z_tgt) + λ · SIGReg(Z)
    
    Anomaly score at inference: s(x) = MSE(z̃_tgt, z_tgt)
```

**LeanEncoder**: 4-layer Transformer with Pre-LN, 4 attention heads, d_model=256, strict causal masking within each segment. No EMA teacher. The same encoder weights process both context and target independently, enforced by causal attention.

**TemporalPredictor**: Three-layer 1D-CNN (kernel=3, channels 256→128→44) with a linear output projection. A convolutional predictor is appropriate here because the 5-minute prediction horizon is localised in time; full attention would be wasteful. The ablation shows +0.024 VUS-PR over a linear mapping under identical conditions.

**SIGReg**: Enforces an isotropic Gaussian on the batch of encoder embeddings by testing random projections via the Epps-Pulley characteristic function test (Balestriero & LeCun, 2025). The Cramér-Wold theorem guarantees that if all 1D projections are standard normal, the full multivariate distribution is isotropic Gaussian. Complexity is O(B²K) — linear in projection count, no architectural changes required. SIGReg decays from 21.82 → 2.47 monotonically over 30 epochs, confirming progressive convergence to the provably optimal embedding geometry without EMA, stop-gradients, or momentum scheduling.

---

## Key results

### Ablation (7 configurations)

| Config | Context/Target | Predictor | λ | Layers | Val MSE | τ | VUS-PR |
|---|---|---|---|---|---|---|---|
| E1 | 10/10 min | DumbPredictor | 0.05 | 2 | 0.2304 | — | — |
| E2 | 10/10 min | DumbPredictor | 0.05 | 4 | 0.2727 | 0.5651 | 0.6668 |
| E3 | 10/10 min | DumbPredictor | 0.10 | 4 | 0.2674 | 0.5757 | 0.6584 |
| E4 | 10/10 min | DumbPredictor | 0.025 | 4 | 0.2382 | 0.4952 | 0.7679 |
| E5 | 30/5 min | DumbPredictor | 0.05 | 4 | 0.0674 | 0.1504 | 0.8525 |
| E6 | 30/5 min | 1D-CNN | 0.05 | 4 | 0.0624 | 0.1681 | 0.8761 |
| **E7** | **30/5 min** | **1D-CNN** | **0.05** | **4** | **0.0254** | **0.0622** | **0.9601** |

The single largest gain (+0.186 VUS-PR) comes from extending context from 10 to 30 minutes; long enough to observe SWaT's inter-stage dynamics. Predictor architecture contributes +0.024. Both effects are independent and additive.

### Computational profile

| Property | Value |
|---|---|
| Total parameters | 1,238,104 |
| Encoder parameters | 810,456 |
| Predictor parameters | 427,648 |
| Training time | ~24 min (single T4 GPU) |
| Peak VRAM | 3,389 MB |
| Inference | Single forward pass, CPU-viable |
| Edge targets | NVIDIA Jetson Nano (4GB), Raspberry Pi 5 (8GB) |

---

## Dataset

**SWaT (Secure Water Treatment)**: iTrust Centre, Singapore University of Technology and Design.

- 51 sensor and actuator channels (44 active after static column removal), 1-second sampling
- ~496,800 timesteps of attack-free training data
- ~449,919 timesteps of test data with 41 attack scenarios (~11.9% anomalous)
- Attacks include pump state manipulation, sensor spoofing, and coordinated multi-stage intrusions

Access requires signing a data usage agreement with iTrust: [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

---

## Preprocessing

1. Drop 7 static/redundant columns: MV101, AIT201, MV201, P201, P202, P204, MV303
2. Downsample 5× (1s → 25s resolution)
3. Apply per-channel RevIN normalization per window; no global statistics, robust to setpoint shifts
4. Because inference is purely latent, the RevIN denormalization step is bypassed

---

## Theoretical grounding

TS-LeJEPA builds directly on two papers:

**LeCun (2022)**: *A Path Towards Autonomous Machine Intelligence.* Argues that world models should predict in representation space, not pixel/sensor space. Unpredictable noise cannot be predicted and so contributes no gradient signal; the encoder learns to discard it.

**Balestriero & LeCun (2025)**: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* Proves via OLS and kernel regression that the isotropic Gaussian is the unique embedding distribution minimising both bias and variance in downstream prediction tasks. Introduces SIGReg to enforce this distribution via the Cramér-Wold theorem and Epps-Pulley characteristic function testing. Validates across 60+ architectures on ImageNet-1k with 3× fewer training epochs than I-JEPA.

TS-LeJEPA applies this framework to ICS anomaly detection. To our knowledge, this is the first LeJEPA application to industrial cyber-physical systems.

---

## Limitations

- Single benchmark (SWaT). WADI, BATADAL, and power grid datasets are the natural next evaluation targets.
- 5× downsampling attenuates sub-25-second attack vectors. Short-duration spikes may be partially missed.
- Precision at τ = 0.0622 is 0.787. Process mode transitions (pump starts, stage switchovers) produce genuine latent transients that appear as false positives. A mode-aware threshold schedule would reduce these.
- No online recalibration. The threshold is locked at the 99th percentile of training scores after epoch 30.

---

## Contact

Chidera Elijah Achinike | MTech AI, Vivekananda Global University  
[LinkedIn](https://linkedin.com/in/aichidera) · [Blog](https://dera-unhinged.hashnode.dev)
