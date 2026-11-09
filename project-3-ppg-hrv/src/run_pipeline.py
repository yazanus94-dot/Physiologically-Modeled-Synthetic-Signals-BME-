"""
End-to-end PPG HRV pipeline:

generate -> detect systolic peaks -> score detector -> HRV metrics for
reference vs detected tachograms -> plots + metrics.json in outputs/.

Run:  cd src && python run_pipeline.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_ppg import generate_ppg
from peak_detect import detect_peaks, score_detector
from hrv_metrics import (time_domain_metrics, _frequency_domain_metrics_from_peaks)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1. Generating 3 min synthetic PPG at 100 Hz (RSA-modulated HRV "
          "+ motion-artifact bursts)...")
    ppg, t, peaks_true, fs = generate_ppg(duration_s=180, fs=100, seed=7)
    print(f"   {len(peaks_true)} true systolic peaks")

    print("2. Detecting systolic peaks (bandpass 0.5-8 Hz + adaptive threshold "
          "+ plausibility filter)...")
    peaks_det = detect_peaks(ppg, fs)
    det_score = score_detector(peaks_det, peaks_true, fs, tolerance_s=0.15)
    print(f"   sensitivity={det_score['sensitivity']:.4f}  "
          f"precision={det_score['precision']:.4f}  "
          f"(TP={det_score['tp']} FP={det_score['fp']} FN={det_score['fn']})")

    print("3. Computing HRV metrics for reference vs detected tachograms...")
    ref_rr_ms = np.diff(peaks_true) / fs * 1000.0
    det_rr_ms = np.diff(peaks_det) / fs * 1000.0
    ref_time = time_domain_metrics(ref_rr_ms)
    det_time = time_domain_metrics(det_rr_ms)
    ref_freq = _frequency_domain_metrics_from_peaks(peaks_true, fs)
    det_freq = _frequency_domain_metrics_from_peaks(peaks_det, fs)

    print("\n  time domain     reference    detected")
    for k in ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct"]:
        print(f"  {k:16s}  {ref_time[k]:>10.2f}  {det_time[k]:>10.2f}")
    print("\n  frequency domain  reference    detected")
    for k in ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio"]:
        print(f"  {k:16s}  {ref_freq[k]:>10.3f}  {det_freq[k]:>10.3f}")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "peak_detection": det_score,
            "n_true_peaks": len(peaks_true),
            "n_detected_peaks": len(peaks_det),
            "hrv_reference": {"time": ref_time, "freq": ref_freq},
            "hrv_detected": {"time": det_time, "freq": det_freq},
        }, f, indent=2)
    print("\nSaved metrics -> outputs/metrics.json")

    # --- Plot 1: PPG strip with detected peaks ---
    win = (25, 35)  # seconds; includes a motion-artifact burst region (at 25% = 45s? -> use 25 s window)
    i0, i1 = int(win[0] * fs), int(win[1] * fs)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t[i0:i1], ppg[i0:i1], color="#1B3A6B", lw=0.8)
    det = peaks_det[(peaks_det >= i0) & (peaks_det < i1)]
    tru = peaks_true[(peaks_true >= i0) & (peaks_true < i1)]
    ax.plot(t[det], ppg[det], "g^", ms=9, label="detected peaks")
    ax.plot(t[tru], ppg[tru] - 0.07, "xr", ms=8, mew=1.5, label="ground truth")
    ax.set_xlabel("Time (s)")
    ax.set_title("Synthetic PPG with detected systolic peaks\n"
                 f"detection sensitivity={det_score['sensitivity']:.3f}, "
                 f"precision={det_score['precision']:.3f}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ppg_with_detected_peaks.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: RR tachogram, detected vs reference ---
    t_ref = (peaks_true[:-1] + peaks_true[1:]) / 2.0 / fs
    t_det = (peaks_det[:-1] + peaks_det[1:]) / 2.0 / fs
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t_ref, ref_rr_ms, color="#2563AE", lw=1.4, label="reference (ground truth)")
    ax.plot(t_det, det_rr_ms, ".", color="#C0392B", ms=3, alpha=0.6,
            label="from detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pulse interval (ms)")
    ax.set_title("RR tachogram — reference vs detected\n"
                 "(spurious interval spikes during motion-artifact bursts inflate "
                 "SDNN/RMSSD)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rr_tachogram.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Welch PSD of the detected tachogram with LF/HF bands ---
    from scipy.signal import welch
    from scipy.interpolate import CubicSpline
    itis_s = det_rr_ms / 1000.0
    t_mid = t_det
    cs = CubicSpline(t_mid, itis_s)
    t_uniform = np.arange(t_mid[0], t_mid[-1], 0.25)
    freqs, psd = welch(cs(t_uniform) - np.mean(cs(t_uniform)),
                       fs=4.0, nperseg=min(256, len(t_uniform)))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(freqs, psd, color="#27AE60")
    for band, lo, hi, col in [("VLF", 0.003, 0.04, "#B8B8B8"),
                              ("LF", 0.04, 0.15, "#F1C40F"),
                              ("HF", 0.15, 0.40, "#E67E22")]:
        p = psd[(freqs >= lo) & (freqs <= hi)]
        f = freqs[(freqs >= lo) & (freqs <= hi)]
        ax.fill_between(f, p, color=col, alpha=0.4, label=f"{band}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (ms^2/Hz)")
    ax.set_title("Welch PSD of the detected RR tachogram (4 Hz)")
    ax.set_xlim(0, 0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "psd_lf_hf.png"), dpi=150)
    plt.close(fig)

    print("Saved plots -> outputs/ppg_with_detected_peaks.png, rr_tachogram.png, "
          "psd_lf_hf.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
