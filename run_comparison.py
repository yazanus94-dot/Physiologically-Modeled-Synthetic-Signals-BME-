"""
Compares denoising methods against a known-clean synthetic ECG reference,
reports SNR improvement + RMSE per method, and plots the results.

Run: python src/run_comparison.py
Output goes to ./outputs/
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_noisy_signal import make_clean_and_noisy
from denoise_methods import (
    median_filter_baseline_removal, bandpass_filter, notch_filter,
    fft_soft_threshold_denoise, combined_pipeline, snr_db, rmse,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1. Generating 30 s clean synthetic ECG + noisy version "
          "(baseline wander + 50 Hz powerline + EMG noise)...")
    clean, noisy, fs, r_peaks, labels, components = make_clean_and_noisy(duration_s=30, fs=250, seed=3)

    print("2. Running each denoising method...")
    median_only, _ = median_filter_baseline_removal(noisy, fs)
    bandpass_only = bandpass_filter(noisy, fs)
    notch_only = notch_filter(noisy, fs)
    fft_denoised = fft_soft_threshold_denoise(noisy, fs)
    combined = combined_pipeline(noisy, fs)

    methods = {
        "raw_noisy": noisy,
        "median_filter_baseline_removal": median_only,
        "bandpass_0.5_40hz": bandpass_only,
        "notch_50hz": notch_only,
        "fft_soft_threshold": fft_denoised,
        "combined_pipeline": combined,
    }

    print("3. Scoring each method against the clean reference (SNR, RMSE)...")
    results = {}
    baseline_snr = snr_db(clean, noisy)
    for name, est in methods.items():
        results[name] = {
            "snr_db": round(snr_db(clean, est), 3),
            "snr_improvement_db": round(snr_db(clean, est) - baseline_snr, 3),
            "rmse": round(rmse(clean, est), 5),
        }
        print(f"   {name:32s}  SNR={results[name]['snr_db']:7.3f} dB "
              f"(+{results[name]['snr_improvement_db']:6.3f} dB)   RMSE={results[name]['rmse']:.5f}")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics -> outputs/metrics.json")

    # --- Plot 1: time-domain comparison, 4 s window ---
    window_s = 4
    n_show = int(window_s * fs)
    t = np.arange(n_show) / fs

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, clean[:n_show], color="#1B3A6B")
    axes[0].set_title("Clean synthetic ECG (ground truth)")
    axes[1].plot(t, noisy[:n_show], color="#C0392B")
    axes[1].set_title(f"Noisy (baseline wander + 50 Hz powerline + EMG)  "
                       f"SNR={results['raw_noisy']['snr_db']:.1f} dB")
    axes[2].plot(t, combined[:n_show], color="#2563AE")
    axes[2].set_title(f"Combined pipeline (median-filter baseline removal -> "
                       f"notch -> bandpass)  SNR={results['combined_pipeline']['snr_db']:.1f} dB")
    axes[3].plot(t, fft_denoised[:n_show], color="#27AE60")
    axes[3].set_title(f"FFT soft-threshold denoiser  SNR={results['fft_soft_threshold']['snr_db']:.1f} dB")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_ylabel("mV (arb.)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "denoising_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: SNR improvement bar chart ---
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [n for n in methods if n != "raw_noisy"]
    vals = [results[n]["snr_improvement_db"] for n in names]
    order = np.argsort(vals)
    ax.barh([names[i].replace("_", " ") for i in order], [vals[i] for i in order], color="#2563AE")
    ax.set_xlabel("SNR improvement over raw noisy signal (dB)")
    ax.set_title("Denoising method comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "snr_improvement.png"), dpi=150)
    plt.close(fig)

    print("Saved plots -> outputs/denoising_comparison.png, snr_improvement.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
