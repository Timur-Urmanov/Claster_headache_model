# Architecture Documentation

## System Overview

The model implements a **hybrid neurophysiological approach** combining:
- Deep learning (encoders, LSTM)
- Dynamical systems ( coupled oscillators)
- Clinical knowledge (hysteresis, circadian rhythms)

## Design Philosophy

**Why this architecture?**

1. **Encoders vs Raw Data:** Compression reduces noise, extracts latent factors
2. **Oscillators vs End-to-End:** Explicit modeling of hypothalamus/PAG/TNC ensures
   predictions are grounded in neurophysiology, not just pattern matching
3. **Global State (z, θ, s):** Interpretable variables clinicians can monitor
4. **Bounded Dynamics:** Tanh/Softsign prevent unrealistic predictions

## Key Design Decisions

### Hysteresis in z

The ictogenicity variable `z` has asymmetric evolution:
- If z > 0.52: accelerates up (dz/dt × 1.5-2.0)
- If z < 0.38: decelerates (dz/dt × 0.3)

**Biological basis:** Once triggered, cluster periods have momentum due to:
- Central sensitization accumulated over days
- Hypothalamic hypersensitivity
- Circadian entrainment

### Phase Wraparound for θ

Circadian phase uses sin/cos encoding in loss function to handle 2π discontinuity:
```python
loss_theta = MSE(sin(θ), sin(θ_target)) + MSE(cos(θ), cos(θ_target))
```

Without this, model sees 0.1 rad and 6.2 rad as far apart, when they are close.

### Masked Biomarkers

Real-world biomarkers are sparse (<5% coverage). Instead of imputation:
```python
biomarker_value * mask
```
When mask=False, gradient doesn\'t flow through missing data.

## Failure Modes (To Monitor)

1. **z Collapse:** If z always stays near 0, model fails to predict
   - Check: latent_variance in Phase 2 training
   - Fix: Increase epsilon (learning rate of z)

2. **Circadian Drift:** If theta doesn\'t maintain 24h period
   - Check: autocorrelation of theta at lag=24
   - Fix: Strengthen circadian regularization

3. **Pain Decoupling:** If z and pain diary uncorrelated in Phase 3
   - Check: Pearson correlation metric
   - Fix: May indicate wrong causal direction (biomarkers as input vs output)

## Performance Targets

**Clinical Utility Thresholds:**
- Time-to-event error: < 2 days (for 7-day prediction)
- Period detection: 80% precision, 90% recall
- False alarm rate: < 1/month (avoid alarm fatigue)

**Current Status:**
- Architecture: ✓ Validated on synthetic data
- Training: Awaiting real clinical data
- Validation: Need 10+ patients with 90+ days each
