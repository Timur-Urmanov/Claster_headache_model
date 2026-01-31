"""
Example: Using the Cluster Headache Model

This script demonstrates:
1. Creating synthetic patient data
2. Running inference
3. Interpreting results for clinical use
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from src.model.model import ClusterHeadacheModel
from src.data.dataset import PatientRecord, PhysiologicalSignals


def create_example_patient(days=30):
    """Create synthetic patient for demonstration"""
    T = days * 24
    start = datetime(2024, 1, 1)
    timestamps = np.array([start + timedelta(hours=i) for i in range(T)])

    return {
        'timestamp': timestamps,
        'time_of_day': np.arange(T) % 24,
        'spo2': np.random.normal(97, 1, T).clip(90, 100),
        'eeg': torch.randn(1, 8, T),
        'hr': torch.randn(1, T) * 10 + 70,
        'hrv_stats': torch.randn(1, 3),
    }


def main():
    print("Cluster Headache Model - Example Usage")
    print("=" * 60)

    # Initialize model
    model = ClusterHeadacheModel()
    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Create example signals
    patient_data = create_example_patient(days=7)  # 1 week

    signals = {
        'eeg': patient_data['eeg'],
        'hr': patient_data['hr'],
        'hrv_stats': patient_data['hrv_stats'],
        'spo2': torch.tensor(patient_data['spo2']).float().unsqueeze(0),
        'cgrp': torch.tensor([200.0]),
        'cortisol': torch.tensor([400.0]),
        'orexin': torch.tensor([150.0]),
        'cgrp_mask': torch.tensor([True]),
        'cortisol_mask': torch.tensor([True]),
        'orexin_mask': torch.tensor([True]),
    }

    print(f"✓ Created signals: {patient_data['eeg'].shape[2]} hours of data")

    # Run inference
    with torch.no_grad():
        outputs = model(signals, num_steps=168, dt=1.0)

    # Results
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    prob_current = outputs['period_predictions']['prob_current'].item()
    prob_start = outputs['period_predictions']['prob_start'].item()
    prob_end = outputs['period_predictions']['prob_end'].item()

    print(f"P(Currently in cluster period): {prob_current:.3f}")
    print(f"P(Will start in next 7 days):   {prob_start:.3f}")
    print(f"P(Will end in next 7 days):     {prob_end:.3f}")

    # Ictogenicity trajectory (critical for safety!)
    z_traj = outputs['trajectory']['z'][0].numpy()
    print(f"\nIctogenicity (z) trajectory:")
    print(f"  Current: {z_traj[-1]:.3f}")
    print(f"  Max: {z_traj.max():.3f}")
    print(f"  Thresholds: A_off=0.38, A_on=0.52")

    if z_traj[-1] > 0.52:
        print("  ⚠️  WARNING: z > A_on - HIGH RISK of cluster period")
    elif z_traj[-1] > 0.38:
        print("  ⚡ CAUTION: z > A_off - ELEVATED RISK")
    else:
        print("  ✓ Stable: z below risk thresholds")

    # Circadian phase
    theta = outputs['trajectory']['theta'][0, -1].item()
    hours = (theta / (2 * np.pi)) * 24
    print(f"\nCircadian phase: {hours:.1f}h (0-24 scale)")

    print("\n" + "=" * 60)
    print("CLINICAL INTERPRETATION")
    print("=" * 60)

    if prob_start > 0.7:
        print("🔴 HIGH PROBILITY: Cluster period likely to start within 7 days")
        print("   Action: Initiate preventive medication, ensure emergency meds available")
        print("   Safety: Check for IAD risk factors, ensure caregiver awareness")
    elif prob_start > 0.3:
        print("🟡 ELEVATED RISK: Possible period start")
        print("   Action: Monitor closely, review abortive meds")
    else:
        print("🟢 LOW RISK: Period unlikely to start immediately")

    print("\nFor IAD emergency protocol, see clinical/protocols.md")


if __name__ == '__main__':
    main()
