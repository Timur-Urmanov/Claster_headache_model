"""
Dataset Specification for Cluster Headache Model

Handles:
- Sparse biomarker measurements (CGRP, Cortisol, Orexin)
- Temporal alignment
- Quality validation
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from datetime import datetime


@dataclass
class PhysiologicalSignals:
    """
    Raw physiological signals
    All time series aligned by timestamp (hourly resolution)
    """
    # Time
    timestamp: np.ndarray  # datetime64, shape (T,)
    time_of_day: np.ndarray  # float [0, 24)
    day_of_year: np.ndarray  # int [1, 365]

    # Core signals (required)
    spo2: np.ndarray  # Oxygen saturation %

    # Optional continuous signals
    eeg_raw: Optional[np.ndarray] = None  # shape (T, n_channels)
    hr: Optional[np.ndarray] = None  # Heart rate
    hrv_rmssd: Optional[np.ndarray] = None
    hrv_lf_hf: Optional[np.ndarray] = None

    # Biomarkers (SPARSE - use masks!)
    cortisol: Optional[np.ndarray] = None
    cortisol_mask: Optional[np.ndarray] = None  # True if measured
    cgrp: Optional[np.ndarray] = None
    cgrp_mask: Optional[np.ndarray] = None
    orexin: Optional[np.ndarray] = None
    orexin_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validation"""
        T = len(self.timestamp)
        assert self.time_of_day.shape == (T,), "time_of_day shape mismatch"
        assert self.spo2.shape == (T,), "spo2 shape mismatch"
        assert np.all((self.time_of_day >= 0) & (self.time_of_day < 24))
        assert np.all((self.spo2 >= 70) & (self.spo2 <= 100))


@dataclass
class SubjectiveData:
    """Pain diary (SPARSE)"""
    timestamp: np.ndarray
    pain_intensity: np.ndarray  # [0, 10]
    pain_mask: np.ndarray  # True if reported
    lacrimation: Optional[np.ndarray] = None
    rhinorrhea: Optional[np.ndarray] = None
    restlessness: Optional[np.ndarray] = None


@dataclass
class Labels:
    """Ground truth for training"""
    timestamp: np.ndarray
    in_cluster_period: np.ndarray  # bool
    period_start_time: Optional[datetime] = None
    period_end_time: Optional[datetime] = None


@dataclass
class PatientRecord:
    """Complete patient record"""
    patient_id: str
    age: int
    sex: str
    cluster_headache_duration_years: int
    signals: PhysiologicalSignals
    subjective: Optional[SubjectiveData] = None
    labels: Optional[Labels] = None
    has_venous_pathology: bool = False
    venous_congestion_score: Optional[float] = None


class DataValidator:
    """Quality control for physiological signals"""

    @staticmethod
    def check_signal_quality(signal: np.ndarray, expected_range: tuple):
        if signal is None or len(signal) == 0:
            return {'valid': False, 'reason': 'empty'}

        min_val, max_val = expected_range
        in_range = np.sum((signal >= min_val) & (signal <= max_val))
        in_range_ratio = in_range / len(signal)

        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        z_scores = np.abs((signal - median) / (mad + 1e-6))
        outlier_ratio = np.sum(z_scores > 3.0) / len(signal)

        return {
            'valid': in_range_ratio > 0.95 and outlier_ratio < 0.05,
            'in_range_ratio': in_range_ratio,
            'outlier_ratio': outlier_ratio
        }
