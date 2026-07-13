"""
Event-based verification metrics for solar wind speed forecast evaluation.

Functions
---------
POD               : Probability of Detection
FNR               : False Negative Rate
PPV               : Positive Predictive Value (Precision)
FAR               : False Alarm Ratio
CSI               : Critical Success Index
BS                : Bias Score
event_verification: Compute all metrics at once and return as a tuple.
"""

import numpy as np


def POD(TP: int, FP: int, FN: int) -> float:
    """Probability of Detection = TP / (TP + FN)"""
    return TP / (TP + FN) if (TP + FN) > 0 else np.nan


def FNR(TP: int, FP: int, FN: int) -> float:
    """False Negative Rate = FN / (TP + FN)"""
    return FN / (TP + FN) if (TP + FN) > 0 else np.nan


def PPV(TP: int, FP: int, FN: int) -> float:
    """Positive Predictive Value = TP / (TP + FP)"""
    return TP / (TP + FP) if (TP + FP) > 0 else np.nan


def FAR(TP: int, FP: int, FN: int) -> float:
    """False Alarm Ratio = FP / (TP + FP)"""
    return FP / (TP + FP) if (TP + FP) > 0 else np.nan


def CSI(TP: int, FP: int, FN: int) -> float:
    """Critical Success Index = TP / (TP + FP + FN)"""
    den = TP + FP + FN
    return TP / den if den > 0 else np.nan


def BS(TP: int, FP: int, FN: int) -> float:
    """Bias Score = (TP + FP) / (TP + FN)"""
    return (TP + FP) / (TP + FN) if (TP + FN) > 0 else np.nan


def event_verification(TP: int, FP: int, FN: int) -> tuple:
    """
    Return (POD, FAR, CSI, BS) rounded to 2 decimal places.
    """
    return (
        np.round(POD(TP, FP, FN), 2),
        np.round(FAR(TP, FP, FN), 2),
        np.round(CSI(TP, FP, FN), 2),
        np.round(BS(TP, FP, FN),  2),
    )
