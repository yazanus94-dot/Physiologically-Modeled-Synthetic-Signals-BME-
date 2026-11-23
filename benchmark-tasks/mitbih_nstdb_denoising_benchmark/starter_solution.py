"""Starter baseline for the MIT-BIH + NSTDB ECG denoising benchmark.

The simplest reasonable solution: a zero-phase Butterworth band-pass
(0.5–40 Hz) on the noisy segment.  Fast, easy to understand, and a
plausible "first attempt" — the reference solution in
``reference_solution.py`` beats it by adding a **median baseline-removal**
step and **50/60 Hz notches**, and by using a **40 Hz low-pass** instead
of the 0.5 Hz high-pass corner: that is worth **+0.90 dB** of mean SNR
improvement and **+0.19** of overall score on the fixed protocol
(measured on real data, 360 cells; see ``expected_results.json``).

The starter loses partly because of the **0.5 Hz high-pass corner**: the
clean MIT-BIH reference still carries real low-frequency ECG content, so
a zero-phase 0.5 Hz high-pass deletes signal as well as baseline wander
(measured 10.8 dB fidelity on the clean signal, versus 28.1 dB for the
40 Hz low-pass alone).

It also fails the evaluator's short-segment edge check, because
``sosfiltfilt`` needs more pad length than the 16-sample probe provides.
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
