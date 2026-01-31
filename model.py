"""
Cluster Headache Period Prediction Model
Hybrid Architecture: Physiological Encoders + Subcortical Oscillators

Author: [Your Name]
License: Apache 2.0 (Open Source Medical Device)
Purpose: Predict cluster headache periods to prevent Ictal Agency Dissociation (IAD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# ENCODERS
# ============================================================================

class EEGEncoder(nn.Module):
    """
    EEG Encoder: raw/spectrum → latent (32-dim)
    Architecture: 1D CNN with Tanh activations (bounded)
    """
    def __init__(self, input_channels=8, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = torch.tanh(self.conv3(h))
        h = self.pool(h).squeeze(-1)
        return torch.tanh(self.fc(h))


class CardioEncoder(nn.Module):
    """Cardio Encoder: HR/HRV → latent (16-dim)"""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hrv_fc = nn.Linear(3, 16)
        self.fusion = nn.Linear(64 + 16, latent_dim)

    def forward(self, hr, hrv_stats):
        hr = hr.unsqueeze(1)
        h = torch.tanh(self.conv1(hr))
        h = torch.tanh(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        hrv = torch.tanh(self.hrv_fc(hrv_stats))
        combined = torch.cat([h, hrv], dim=-1)
        return torch.tanh(self.fusion(combined))


class SpO2Encoder(nn.Module):
    """SpO2 Encoder with trend extraction (8-dim)"""
    def __init__(self, latent_dim=8):
        super().__init__()
        self.trend_conv = nn.Conv1d(1, 16, kernel_size=31, padding=15)
        self.fast_conv = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, latent_dim)

    def forward(self, spo2):
        spo2 = spo2.unsqueeze(1)
        trend = torch.tanh(self.trend_conv(spo2))
        fast = torch.tanh(self.fast_conv(spo2))
        h = torch.cat([trend, fast], dim=1)
        h = self.pool(h).squeeze(-1)
        return torch.tanh(self.fc(h))


class BiomarkerEncoder(nn.Module):
    """
    Biomarker Encoder with masking for missing values
    Handles: CGRP, Cortisol, Orexin (sparse measurements)
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.cgrp_proj = nn.Linear(1, 8)
        self.cortisol_proj = nn.Linear(1, 8)
        self.orexin_proj = nn.Linear(1, 8)
        self.fusion = nn.Linear(24, latent_dim)

    def forward(self, cgrp, cortisol, orexin, 
                cgrp_mask, cortisol_mask, orexin_mask):
        cgrp_emb = torch.tanh(self.cgrp_proj(cgrp.unsqueeze(-1))) * cgrp_mask.unsqueeze(-1).float()
        cortisol_emb = torch.tanh(self.cortisol_proj(cortisol.unsqueeze(-1))) * cortisol_mask.unsqueeze(-1).float()
        orexin_emb = torch.tanh(self.orexin_proj(orexin.unsqueeze(-1))) * orexin_mask.unsqueeze(-1).float()
        combined = torch.cat([cgrp_emb, cortisol_emb, orexin_emb], dim=-1)
        return torch.tanh(self.fusion(combined))


# ============================================================================
# SUBCORTICAL OSCILLATORS (Bounded Dynamics)
# ============================================================================

class BoundedOscillator(nn.Module):
    """Base oscillator with bounded outputs (Tanh)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = torch.tanh(self.fc_out(out))
        return out, h


class HYPOscillator(BoundedOscillator):
    """Hypothalamus: [arousal, orexin, d_theta]"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__(input_dim, hidden_dim, output_dim=3)


class PAGOscillator(nn.Module):
    """
    PAG with On/Off populations (bistable)
    Wilson-Cowan style dynamics with cross-inhibition
    """
    def __init__(self, input_dim, hidden_dim=28):
        super().__init__()
        self.on_gru = nn.GRU(input_dim, hidden_dim // 2, batch_first=True)
        self.off_gru = nn.GRU(input_dim, hidden_dim // 2, batch_first=True)
        self.fc_on = nn.Linear(hidden_dim // 2, 1)
        self.fc_off = nn.Linear(hidden_dim // 2, 1)
        self.w_inhib = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, h_on=None, h_off=None):
        out_on, h_on = self.on_gru(x, h_on)
        out_off, h_off = self.off_gru(x, h_off)
        on_activity = torch.sigmoid(self.fc_on(out_on))
        off_activity = torch.sigmoid(self.fc_off(out_off))
        on_final = on_activity * torch.sigmoid(-self.w_inhib * off_activity)
        off_final = off_activity * torch.sigmoid(-self.w_inhib * on_activity)
        return on_final, off_final, h_on, h_off


class TNCOscillator(BoundedOscillator):
    """Trigeminal Nucleus: [pain, cgrp, sensitization_delta]"""
    def __init__(self, input_dim, hidden_dim=48):
        super().__init__(input_dim, hidden_dim, output_dim=3)


# ============================================================================
# GLOBAL STATE TRACKER
# ============================================================================

class GlobalStateTracker(nn.Module):
    """
    Global state: (z, theta, sensitization)
    z: Ictogenicity (0 to inf)
    theta: Circadian phase (0 to 2π)
    sensitization: Pain memory (0 to 1)
    """
    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.003))
        self.gamma = nn.Parameter(torch.tensor(0.10))
        self.A_on = nn.Parameter(torch.tensor(0.52))  # Hysteresis threshold
        self.A_off = nn.Parameter(torch.tensor(0.38))
        self.sens_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_prev, theta_prev, sens_prev, 
                hyp_arousal, tnc_pain, dt=1.0):
        # Evolution of z (adaptive with hysteresis)
        z_mean = z_prev.mean()
        if z_mean > self.A_on:
            dz_factor = 1.5 + 0.8 * (z_mean - self.A_on)
        elif z_mean < self.A_off:
            dz_factor = 0.3
        else:
            dz_factor = 1.0

        dz = self.epsilon * dz_factor * (hyp_arousal - self.gamma * z_prev)
        z_new = torch.clamp(z_prev + dz * dt, min=0.0)

        # Evolution of theta (circadian)
        omega_base = 2 * np.pi / 24.0
        d_theta = omega_base * dt * (1 + 0.3 * hyp_arousal)
        theta_new = torch.remainder(theta_prev + d_theta, 2 * np.pi)

        # Evolution of sensitization (plasticity)
        d_sens = 0.01 * (tnc_pain - self.sens_decay * sens_prev)
        sens_new = torch.clamp(sens_prev + d_sens, 0.0, 1.0)

        return z_new, theta_new, sens_new


# ============================================================================
# PERIOD PREDICTOR
# ============================================================================

class PeriodPredictor(nn.Module):
    """
    Predicts cluster period transitions
    Outputs: P(start|7days), P(end|7days), P(current)
    """
    def __init__(self, state_dim=11, history_length=168):
        super().__init__()
        self.history_length = history_length
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc_start = nn.Linear(128, 1)
        self.fc_end = nn.Linear(128, 1)
        self.fc_current = nn.Linear(128, 1)

    def forward(self, state_history):
        out, _ = self.lstm(state_history)
        final_h = out[:, -1, :]
        prob_start = torch.sigmoid(self.fc_start(final_h)).squeeze(-1)
        prob_end = torch.sigmoid(self.fc_end(final_h)).squeeze(-1)
        prob_current = torch.sigmoid(self.fc_current(final_h)).squeeze(-1)
        return {
            'prob_start': prob_start,
            'prob_end': prob_end,
            'prob_current': prob_current,
        }


# ============================================================================
# FULL MODEL
# ============================================================================

class ClusterHeadacheModel(nn.Module):
    """
    End-to-end model for cluster headache period prediction

    Pipeline:
    Raw signals → Encoders → Latent → Oscillators → State → Predictor
    """
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder(latent_dim=32)
        self.cardio_encoder = CardioEncoder(latent_dim=16)
        self.spo2_encoder = SpO2Encoder(latent_dim=8)
        self.biomarker_encoder = BiomarkerEncoder(latent_dim=16)

        self.latent_dim = 72
        self.HYP = HYPOscillator(input_dim=self.latent_dim + 3, hidden_dim=32)
        self.PAG = PAGOscillator(input_dim=self.latent_dim + 3, hidden_dim=28)
        self.TNC = TNCOscillator(input_dim=self.latent_dim + 3, hidden_dim=48)

        self.state_tracker = GlobalStateTracker()
        state_dim = 11  # z + theta + sens + node_outputs
        self.period_predictor = PeriodPredictor(state_dim=state_dim)

    def forward(self, signals, num_steps, dt=1.0):
        batch_size = signals['spo2'].size(0)
        device = signals['spo2'].device

        # Encoding
        eeg_latent = self.eeg_encoder(signals['eeg'])
        cardio_latent = self.cardio_encoder(signals['hr'], signals['hrv_stats'])
        spo2_latent = self.spo2_encoder(signals['spo2'])
        bio_latent = self.biomarker_encoder(
            signals['cgrp'], signals['cortisol'], signals['orexin'],
            signals['cgrp_mask'], signals['cortisol_mask'], signals['orexin_mask']
        )
        latent = torch.cat([eeg_latent, cardio_latent, spo2_latent, bio_latent], dim=-1)

        # Initialization
        z = torch.zeros(batch_size, device=device)
        theta = torch.zeros(batch_size, device=device)
        sens = torch.zeros(batch_size, device=device)
        h_hyp = h_pag_on = h_pag_off = h_tnc = None

        trajectory = {k: [] for k in ['z', 'theta', 'sens', 'HYP_arousal', 'HYP_orexin',
                                       'PAG_on', 'PAG_off', 'TNC_pain', 'TNC_cgrp']}
        state_history = []

        # Temporal simulation
        for t in range(num_steps):
            osc_input = torch.cat([latent, z.unsqueeze(-1), theta.unsqueeze(-1), sens.unsqueeze(-1)], dim=-1).unsqueeze(1)

            hyp_out, h_hyp = self.HYP(osc_input, h_hyp)
            hyp_out = hyp_out.squeeze(1)

            pag_on, pag_off, h_pag_on, h_pag_off = self.PAG(osc_input, h_pag_on, h_pag_off)
            pag_on = pag_on.squeeze(1)
            pag_off = pag_off.squeeze(1)

            tnc_out, h_tnc = self.TNC(osc_input, h_tnc)
            tnc_out = tnc_out.squeeze(1)

            z, theta, sens = self.state_tracker(z, theta, sens, hyp_out[:, 0], tnc_out[:, 0], dt)

            trajectory['z'].append(z)
            trajectory['theta'].append(theta)
            trajectory['sens'].append(sens)
            trajectory['HYP_arousal'].append(hyp_out[:, 0])
            trajectory['HYP_orexin'].append(hyp_out[:, 1])
            trajectory['PAG_on'].append(pag_on.squeeze(-1))
            trajectory['PAG_off'].append(pag_off.squeeze(-1))
            trajectory['TNC_pain'].append(tnc_out[:, 0])
            trajectory['TNC_cgrp'].append(tnc_out[:, 1])

            state_t = torch.stack([z, theta, sens, hyp_out[:, 0], hyp_out[:, 1], hyp_out[:, 2],
                                  pag_on.squeeze(-1), pag_off.squeeze(-1),
                                  tnc_out[:, 0], tnc_out[:, 1], tnc_out[:, 2]], dim=-1)
            state_history.append(state_t)

        for key in trajectory:
            trajectory[key] = torch.stack(trajectory[key], dim=1)
        state_history = torch.stack(state_history, dim=1)

        # Predictions
        if num_steps >= 168:
            state_window = state_history[:, -168:, :]
        else:
            pad_len = 168 - num_steps
            padding = torch.zeros(batch_size, pad_len, state_history.size(-1), device=device)
            state_window = torch.cat([padding, state_history], dim=1)

        period_predictions = self.period_predictor(state_window)

        return {
            'trajectory': trajectory,
            'period_predictions': period_predictions,
            'state_history': state_history,
            'latent': latent,
        }


if __name__ == '__main__':
    # Test
    model = ClusterHeadacheModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
