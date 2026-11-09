"""Fast unit tests for the generator and denoising methods."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_noisy_signal import make_clean_and_noisy  # noqa: E402
from denoise_methods import (bandpass_filter, notch_filter,  # noqa: E402
                             combined_pipeline, snr_db, rmse)


def test_generation_shapes():
    clean, noisy, fs, r_peaks, labels, comps = make_clean_and_noisy(
        duration_s=10, fs=250, seed=3)
    # chunk placement can run slightly past the target duration — the
    # generator is the user's original code, kept as-is
    assert len(clean) == len(noisy) >= 10 * 250
    assert fs == 250
    assert len(r_peaks) == len(labels)


def test_bandpass_improves_snr():
    clean, noisy, fs, *_ = make_clean_and_noisy(duration_s=10, fs=250, seed=3)
    assert snr_db(clean, noisy) < 0
    for est in [
        bandpass_filter(noisy, fs),
        notch_filter(noisy, fs),
        combined_pipeline(noisy, fs),
    ]:
        assert snr_db(clean, est) > snr_db(clean, noisy)
        assert rmse(clean, est) < rmse(clean, noisy)
