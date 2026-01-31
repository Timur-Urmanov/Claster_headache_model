"""
4-Phase Training Strategy for Cluster Headache Model

Phase 1: Synthetic Pretraining
Phase 2: Self-Supervised (no labels)
Phase 3: Weak Supervision (pain diary)
Phase 4: Period Prediction Fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict


class SyntheticPretrainingLoss(nn.Module):
    """Phase 1: Learn stable dynamics on synthetic data"""
    def __init__(self, lambda_state=1.0, lambda_bio=1.0, lambda_dynamics=0.5):
        super().__init__()
        self.lambda_state = lambda_state
        self.lambda_bio = lambda_bio
        self.lambda_dynamics = lambda_dynamics

    def forward(self, predicted, targets):
        # State reconstruction
        loss_z = F.mse_loss(predicted['trajectory']['z'], targets['z_true'])

        # Phase wraparound for theta (critical for circadian!)
        loss_theta = (F.mse_loss(torch.sin(predicted['trajectory']['theta']), 
                                 torch.sin(targets['theta_true'])) +
                     F.mse_loss(torch.cos(predicted['trajectory']['theta']), 
                                torch.cos(targets['theta_true'])))

        loss_sens = F.mse_loss(predicted['trajectory']['sens'], targets['sensitization_true'])
        loss_state = loss_z + loss_theta + loss_sens

        # Biomarker reconstruction
        loss_cgrp = F.mse_loss(predicted['trajectory']['TNC_cgrp'], targets['cgrp_true'])
        loss_orexin = F.mse_loss(predicted['trajectory']['HYP_orexin'], targets['orexin_true'])
        loss_bio = loss_cgrp + loss_orexin

        # Smoothness regularization (prevent unrealistic jumps)
        def smoothness_loss(signal):
            diff2 = signal[:, 2:] - 2 * signal[:, 1:-1] + signal[:, :-2]
            return (diff2 ** 2).mean()

        loss_smooth_z = smoothness_loss(predicted['trajectory']['z'])
        loss_smooth_pain = smoothness_loss(predicted['trajectory']['TNC_pain'])
        loss_dynamics = loss_smooth_z + loss_smooth_pain

        total_loss = (self.lambda_state * loss_state + 
                     self.lambda_bio * loss_bio + 
                     self.lambda_dynamics * loss_dynamics)

        return total_loss, {
            'state': loss_state.item(),
            'biomarker': loss_bio.item(), 
            'dynamics': loss_dynamics.item(),
            'total': total_loss.item()
        }


class WeakSupervisionLoss(nn.Module):
    """
    Phase 3: Learn from sparse pain diary

    CRITICAL: Pain diary has only ~5-10% coverage!
    Uses masked loss to ignore missing entries.
    """
    def __init__(self, lambda_pain=1.0, lambda_aux=0.3):
        super().__init__()
        self.lambda_pain = lambda_pain
        self.lambda_aux = lambda_aux

    def forward(self, predicted, pain_diary, pain_mask):
        # Pain reconstruction (scale [0,1] to [0,10])
        pred_pain = predicted['trajectory']['TNC_pain'] * 10

        # MASKED MSE - only where pain was reported!
        pain_error = (pred_pain - pain_diary) ** 2
        pain_error_masked = pain_error * pain_mask.float()
        loss_pain = pain_error_masked.sum() / (pain_mask.sum() + 1e-6)

        # Auxiliary: correlation between z and pain
        z = predicted['trajectory']['z']
        pain_norm = pain_diary / 10.0

        z_mean, pain_mean = z.mean(), pain_norm.mean()
        cov = ((z - z_mean) * (pain_norm - pain_mean)).mean()
        corr = cov / ((z.std() + 1e-6) * (pain_norm.std() + 1e-6))
        loss_aux = -corr

        total_loss = self.lambda_pain * loss_pain + self.lambda_aux * loss_aux

        return total_loss, {
            'pain': loss_pain.item(),
            'correlation': corr.item(),
            'total': total_loss.item()
        }


class PeriodPredictionLoss(nn.Module):
    """
    Phase 4: Predict period transitions

    Predicts:
    - P(currently in cluster period)
    - P(will start in next 7 days)
    - P(will end in next 7 days)
    """
    def __init__(self, lambda_start=1.0, lambda_end=1.0, lambda_current=0.5, time_horizon_days=7):
        super().__init__()
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_current = lambda_current
        self.time_horizon = time_horizon_days * 24  # hours

    def forward(self, predicted, labels):
        # Current state
        pred_current = predicted['period_predictions']['prob_current']
        target_current = labels['in_cluster_period'][:, -1].float()
        loss_current = F.binary_cross_entropy(pred_current, target_current)

        # Start prediction (next N days)
        pred_start = predicted['period_predictions']['prob_start']
        currently_out = ~labels['in_cluster_period'][:, -1]
        future = labels['in_cluster_period'][:, -self.time_horizon:]
        will_start = currently_out & future.any(dim=1)
        loss_start = F.binary_cross_entropy(pred_start, will_start.float())

        # End prediction
        pred_end = predicted['period_predictions']['prob_end']
        currently_in = labels['in_cluster_period'][:, -1]
        will_end = currently_in & (~future.all(dim=1))
        loss_end = F.binary_cross_entropy(pred_end, will_end.float())

        total_loss = (self.lambda_current * loss_current + 
                     self.lambda_start * loss_start + 
                     self.lambda_end * loss_end)

        return total_loss, {
            'current': loss_current.item(),
            'start': loss_start.item(),
            'end': loss_end.item(),
            'total': total_loss.item()
        }


class MultiPhaseTrainer:
    """Trainer managing all 4 phases"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        self.loss_synthetic = SyntheticPretrainingLoss()
        self.loss_weak = WeakSupervisionLoss()
        self.loss_period = PeriodPredictionLoss()

    def train_phase(self, phase_name, dataset, loss_fn, num_epochs, lr=1e-3):
        """Generic training phase"""
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_name}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []

            for batch in dataset:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                outputs = self.model(batch['signals'], num_steps=batch['num_steps'])

                if phase_name == "Synthetic Pretraining":
                    loss, _ = loss_fn(outputs, batch['targets'])
                elif phase_name == "Weak Supervision":
                    loss, _ = loss_fn(outputs, batch['pain_diary'], batch['pain_mask'])
                elif phase_name == "Period Prediction":
                    loss, _ = loss_fn(outputs, batch['labels'])
                else:
                    loss = torch.tensor(0.0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")

        print(f"Phase {phase_name} complete!")
