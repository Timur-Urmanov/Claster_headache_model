# Cluster Headache Period Prediction Model

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open Source](https://img.shields.io/badge/Open%20Source-Medical%20Device-green)]()

**Prediction of cluster headache periods to prevent Ictal Agency Dissociation (IAD) and reduce suicide risk.**

This is an open-source neurophysiological model designed to predict the onset of cluster headache periods 7-14 days in advance, enabling preventive treatment and safety protocols.

> **Status:** Architecture complete, awaiting clinical data for training  
> **License:** Apache 2.0 (free for medical use, no patents)  
> **Author:** [Your Name]  
> **Purpose:** Prevent suffering, not profit

---

## Why This Exists

Cluster headache (CH) has the highest suicide risk of any pain condition. During attacks lasting >60 minutes, patients may experience **Ictal Agency Dissociation (IAD)**—retaining consciousness but losing behavioral control.

**Current problems:**
- Patients hide symptoms due to stigma ("it\'s just a headache")
- Demoralization (OR≈6.66 for suicide) is missed because it\'s not depression
- No early warning system for cluster periods

**This solution:**
- Predicts periods using physiology (EEG, HRV, SpO2, biomarkers)
- Provides 7-14 day warning for preventive medication
- Flags IAD risk for caregiver intervention

---

## Architecture

```
Raw Signals → Encoders → Latent → Oscillators → State → Prediction
```

### Components

**1. Encoders (338K params total)**
- `EEGEncoder`: 1D-CNN on EEG raw/spectrum (32-dim)
- `CardioEncoder`: HR/HRV patterns (16-dim)
- `SpO2Encoder`: Trend + fast components (8-dim)
- `BiomarkerEncoder`: CGRP/Cortisol/Orexin with masking (16-dim)

**2. Subcortical Oscillators**
- `HYPOscillator`: Hypothalamus, generates circadian arousal
- `PAGOscillator`: Periaqueductal gray, On/Off bistable populations
- `TNCOscillator`: Trigeminal nucleus, pain and CGRP release

**3. Global State Tracker**
- `z`: Ictogenicity (0→∞), hysteresis thresholds at 0.38/0.52
- `theta`: Circadian phase (0-2π)
- `sensitization`: Pain memory/plasticity (0-1)

**4. Period Predictor**
- LSTM on 7-day history (168 hours)
- Outputs: P(current period), P(start|7d), P(end|7d)

---

## Training Strategy

### Phase 1: Synthetic Pretraining (50 epochs)
- Full simulation targets (z, θ, sensitization, biomarkers)
- Loss: MSE(state) + MSE(biomarker) + smoothness regularization
- Goal: Stable oscillator dynamics

### Phase 2: Self-Supervised (30 epochs)
- Real signals, NO labels
- Variance preservation + cross-modal consistency
- Goal: Adapt to real-world data distribution

### Phase 3: Weak Supervision (20 epochs)
- Pain diary (sparse ~5-10% coverage!)
- **Masked loss**: Only calculate MSE where diary exists
- Correlation: Force z to correlate with pain
- Goal: Link latent state to subjective pain

### Phase 4: Period Prediction (30 epochs)
- Rare event labels (period start/end)
- Binary cross-entropy on 3 heads
- Goal: Clinical prediction

---

## Data Requirements

### Minimum Viable Dataset
- **Duration:** 90 days continuous per patient
- **Resolution:** Hourly for core signals
- **Signals:**
  - Required: SpO2 (100% coverage)
  - Recommended: EEG, HR/HRV
  - Optional: CGRP, Cortisol, Orexin (sparse okay!)
- **Labels:** Period start/end dates
- **Diary:** Pain intensity (even 5% coverage helps)

### Data Format
```python
from src.data.dataset import PatientRecord, PhysiologicalSignals

record = PatientRecord(
    patient_id="patient_001",
    age=45,
    sex="M",
    cluster_headache_duration_years=10,
    signals=PhysiologicalSignals(
        timestamp=...,  # datetime64
        time_of_day=...,  # [0, 24)
        spo2=...,  # %
        hr=...,  # optional
        cortisol=...,  # optional, sparse
        cortisol_mask=...,  # bool, True if measured
    ),
    has_venous_pathology=True,  # If applicable
)
```

---

## Clinical Protocols

See [clinical/protocols.md](clinical/protocols.md)

### Critical: Ictal Agency Dissociation (IAD)

**Trigger:** Attack duration >60 minutes

**Signs:**
- Patient conscious but cannot control actions/speech
- May appear "uncooperative" but is dissociated
- Cannot self-administer medication

**Action:**
1. Physical safety (remove harmful objects)
2. Administer medication without waiting for explicit consent
3. Stay calm (patient remembers everything)
4. Monitor for 30 min post-attack

### Suicide Risk Warning

**Demoralization ≠ Depression**
- Hopelessness, entrapment, cognitive rigidity
- **Odds Ratio 6.66** for suicidality
- Check [clinical/protocols.md](clinical/protocols.md) for screening

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cluster-headache-model.git
cd cluster-headache-model
pip install -r requirements.txt
```

### Dependencies
- torch >= 2.0
- numpy
- (Optional) pyarrow for data storage

---

## Usage

### Quick Start

```python
import torch
from src.model.model import ClusterHeadacheModel

# Initialize
model = ClusterHeadacheModel()

# Prepare signals (batch_size=1, time=168 hours)
signals = {
    'eeg': torch.randn(1, 8, 168),  # (batch, channels, time)
    'hr': torch.randn(1, 168) * 10 + 70,
    'hrv_stats': torch.randn(1, 3),
    'spo2': torch.randn(1, 168) * 2 + 97,
    'cgrp': torch.randn(1) * 50 + 200,
    'cortisol': torch.randn(1) * 100 + 400,
    'orexin': torch.randn(1) * 50 + 150,
    'cgrp_mask': torch.tensor([True]),
    'cortisol_mask': torch.tensor([True]),
    'orexin_mask': torch.tensor([True]),
}

# Forward pass
with torch.no_grad():
    outputs = model(signals, num_steps=168)

# Check predictions
print(f"P(current period): {outputs['period_predictions']['prob_current']:.3f}")
print(f"P(start in 7d): {outputs['period_predictions']['prob_start']:.3f}")
print(f"P(end in 7d): {outputs['period_predictions']['prob_end']:.3f}")

# Monitor ictogenicity
z_trajectory = outputs['trajectory']['z']  # Check if approaching 0.52
```

### Training

```python
from src.training.trainer import MultiPhaseTrainer

trainer = MultiPhaseTrainer(model, device='cuda')

# Phase 1: Synthetic
trainer.train_phase("Synthetic Pretraining", synthetic_dataset, 
                   trainer.loss_synthetic, num_epochs=50)

# Phase 3: Weak supervision (sparse diary)
trainer.train_phase("Weak Supervision", diary_dataset, 
                   trainer.loss_weak, num_epochs=20)

# Phase 4: Period prediction
trainer.train_phase("Period Prediction", labeled_dataset, 
                   trainer.loss_period, num_epochs=30)
```

---

## Warning & Disclaimer

**This is NOT a medical device (yet).**

- Requires clinical validation before use
- Do not use as sole basis for treatment decisions
- Always maintain human clinical oversight
- Model may exhibit bias specific to training population

**Ethical Use:**
- Never use to deny care ("model says you\'re not in risk")
- Patient has right to override predictions
- Data privacy is paramount (physiological data is identifiable)

---

## Contributing

We need:
1. **Clinical data** (anonymized, 90+ days per patient)
2. **Clinicians** to validate predictions
3. **Patients** for feedback on usability
4. **Developers** for edge case handling

**No commercial interests.** This project exists solely to reduce suffering.

---

## Citation

If you use this model in research:

```bibtex
@software{cluster_headache_model,
  author = [Your Name],
  title = {Cluster Headache Period Prediction Model},
  year = {2024},
  license = {Apache-2.0},
  url = {https://github.com/YOUR_USERNAME/cluster-headache-model}
}
```

---

## Contact

- **Issues:** GitHub Issues (for bugs/technical)
- **Clinical:** [Your Email] (for collaboration)
- **Urgent:** If you are experiencing cluster headache and suicidal thoughts, 
  contact emergency services immediately. This model cannot help you in acute crisis.

---

**Built with the conviction that medical AI should prevent suffering, not extract profit.**
