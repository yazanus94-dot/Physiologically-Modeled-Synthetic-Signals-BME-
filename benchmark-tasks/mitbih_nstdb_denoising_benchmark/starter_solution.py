"""Starter baseline for the MIT-BIH + NSTDB ECG denoising benchmark.

The simplest reasonable solution: a zero-phase Butterworth band-pass
(0.5–40 Hz) on the noisy segment.  Fast, easy to understand, and a
plausible "first attempt" — the reference solution in
``reference_solution.py`` beats it by adding a median-baseline-removal
step and an FFT soft-threshold denoiser.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class DenoisingResult:
    signal: np.ndarray
    method: str


def denoise(noisy, fs):
    """Band-pass filter (0.5–40 Hz, zero-phase, order 3)."""
    sos = butter(3, [0.5, 40.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(noisy, dtype=float))


METHOD_NAME = "bandpass_0.5_40hz"


def starter_solution():
    return _StarterSolution()


class _StarterSolution:
    def denoise(self, noisy, fs):
        return DenoisingResult(signal=denoise(noisy, fs), method=METHOD_NAME)
