"""
Plotting helpers for the real-data (NSTDB) ECG denoising figures.

All helpers save to a path you give them and return the path, so the
experiment runner can compose figures without touching matplotlib state
itself. The backend is forced to Agg (headless-safe).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD_COLORS = {
    "raw_noisy": "#C0392B",
    "median_filter": "#8E44AD",
    "bandpass_0.5_40hz": "#1B3A6B",
    "notch_50hz": "#D68910",
    "fft_soft_threshold": "#27AE60",
    "combined_pipeline": "#2563AE",
}
DEFAULT_COLOR = "#566573"


def _method_color(name):
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    for key, color in METHOD_COLORS.items():
        if key.split("_")[0] in name:
            return color
    return DEFAULT_COLOR


def plot_sample_comparison(clean, noisy, outputs, fs, path,
                           start_s=0.0, window_s=6.0, title=None,
                           achieved_snr_db=None):
    """Stacked waveform panel: clean reference, noisy input, and every
    denoised output over the same time window.

    ``outputs`` is a dict ``{method_name: signal}``.
    """
    start = int(round(start_s * fs))
    n = int(round(window_s * fs))
    stop = min(start + n, len(clean))
    t = (np.arange(start, stop) - start) / fs

    n_rows = 2 + len(outputs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 1.7 * n_rows),
                             sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(t, clean[start:stop], color="#1B3A6B", lw=0.9)
    noise_label = "Noisy input"
    if achieved_snr_db is not None:
        noise_label += f" (SNR {achieved_snr_db:.1f} dB)"
    axes[0].set_title(title or "Real ECG: clean reference vs. denoising "
                               "methods")
    axes[1].plot(t, noisy[start:stop], color="#C0392B", lw=0.9)
    axes[1].set_title(noise_label, fontsize=9)

    for ax, (name, sig) in zip(axes[2:], outputs.items()):
        ax.plot(t, np.asarray(sig)[start:stop],
                color=_method_color(name), lw=0.9)
        ax.set_title(name.replace("_", " "), fontsize=9)

    for ax in axes:
        ax.set_ylabel("mV")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time within segment (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _group_rows(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def plot_metric_by_snr(rows, metric, path, title=None, ylabel=None,
                       noise_keys=("noise_category", "noise_record")):
    """One subplot per noise type; x = target SNR, one line per method."""
    noise_key = next((k for k in noise_keys
                      if rows and k in rows[0]), None)
    if noise_key is None:
        raise ValueError(f"rows carry none of the noise keys {noise_keys}")
    by_noise = _group_rows(rows, noise_key)

    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])

    n_cols = len(by_noise)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.0),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (noise_name, noise_rows) in zip(axes, sorted(by_noise.items())):
        by_snr = _group_rows(noise_rows, "target_snr_db")
        for method in methods:
            xs, ys, yerr = [], [], []
            for snr in sorted(by_snr):
                values = [r[metric] for r in by_snr[snr]
                          if r["method"] == method and
                          np.isfinite(r.get(metric, float("nan")))]
                if values:
                    xs.append(snr)
                    ys.append(float(np.mean(values)))
                    yerr.append(float(np.std(values, ddof=0)))
            if xs:
                ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=3.5, lw=1.4,
                            capsize=2.5, color=_method_color(method),
                            label=method.replace("_", " "))
        ax.set_title(str(noise_name).replace("_", " "))
        ax.set_xlabel("Target SNR (dB)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel or metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(3, len(labels)), fontsize=8, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95 if title else 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_summary(rows, metric, path, title=None, ylabel=None):
    """Mean of ``metric`` across every row, one bar per method.

    Pools all noise types — a compact "what does each method do on
    average" figure distinct from the per-SNR ``plot_metric_by_snr``.
    """
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    means, errs = [], []
    for m in methods:
        vals = [r[metric] for r in rows if r["method"] == m
                and np.isfinite(r.get(metric, float("nan")))]
        means.append(float(np.mean(vals)) if vals else np.nan)
        errs.append(float(np.std(vals, ddof=0)) if vals else np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=errs, capsize=3,
           color=[_method_color(m) for m in methods],
           tick_label=[m.replace("_", " ") for m in methods])
    ax.set_title(title or f"Mean {metric} per method")
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_noise_type_comparison(rows, path, title=None):
    """Mean SNR improvement per noise type at two operating points.

    Two subplots: low SNR (−6 dB) and high SNR (+24 dB).  Each subplot
    shows the mean ``snr_improvement_db`` per noise type (pooled across
    methods) so the *noise-type* dimension is what is being compared.
    """
    def _means(rows_):
        out = {}
        for r in rows_:
            out.setdefault(r["noise_category"], []).append(r["snr_improvement_db"])
        return {k: float(np.mean(v)) for k, v in sorted(out.items())
                if np.isfinite(np.mean(v))}

    low = [r for r in rows if r["target_snr_db"] == -6.0]
    high = [r for r in rows if r["target_snr_db"] == 24.0]
    low_m = _means(low)
    high_m = _means(high)
    cats = sorted(set(low_m) | set(high_m))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, snr, m in ((axes[0], -6.0, low_m), (axes[1], 24.0, high_m)):
        vals = [m.get(c, np.nan) for c in cats]
        ax.bar([c.replace("_", " ") for c in cats], vals,
               color=[_method_color(c) for c in cats] if vals else DEFAULT_COLOR)
        ax.set_title(f"Mean SNR improvement @ {snr:+.0f} dB")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("SNR improvement over raw noisy (dB)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_morphology_summary(rows, path, title=None):
    """Morphology preservation per method.

    Grouped bars of the mean relative change in QRS amplitude and QRS
    width (0 = perfect preservation) across all rows and noise types.
    """
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])

    def _mean(col, m):
        vals = [r[col] for r in rows if r["method"] == m
                and np.isfinite(r[col])]
        return float(np.mean(vals)) if vals else np.nan

    amp = [_mean("qrs_amp_rel_change", m) for m in methods]
    wid = [_mean("qrs_width_rel_change", m) for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    x = np.arange(len(methods))
    w = 0.38
    ax.bar(x - w / 2, amp, w, label="QRS amplitude rel. change",
           color="#1B3A6B", edgecolor="k", alpha=0.8)
    ax.bar(x + w / 2, wid, w, label="QRS width rel. change",
           color="#D68910", edgecolor="k", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in methods], rotation=25,
                       ha="right", fontsize=9)
    ax.set_ylabel("Mean relative change (0 = perfect preservation)")
    ax.set_title(title or "Morphology preservation per method")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_powerline_ablation(rows, path, title=None):
    """Mean SNR improvement (dB) per method on the powerline-ablation
    signal set.

    One subplot per powerline frequency (50 Hz, 60 Hz); each shows the
    mean ``snr_improvement_db`` per method with error bars.
    """
    freqs = sorted({r["freq_hz"] for r in rows})
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])

    n_cols = max(len(freqs), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, f in zip(axes, freqs):
        means, errs = [], []
        for m in methods:
            vals = [r.get("snr_improvement_db", r["snr_db"]) for r in rows
                    if r["freq_hz"] == f and r["method"] == m
                    and np.isfinite(r.get("snr_improvement_db",
                                          r["snr_db"]))]
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(float(np.std(vals, ddof=0)) if vals else np.nan)
        ax.bar(np.arange(len(methods)), means, yerr=errs, capsize=3,
               color=[_method_color(m) for m in methods],
               tick_label=[m.replace("_", " ") for m in methods])
        ax.set_title(f"Powerline @ {int(f)} Hz")
        ax.set_ylabel("SNR improvement over clean ECG (dB)")
        ax.grid(alpha=0.3, axis="y")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tolerance_sensitivity(rows, path, title=None):
    """Mean R-peak sensitivity / precision / F1 per tolerance.

    Bars: mean sensitivity, mean precision, mean F1 at ±50, ±100, ±150 ms.
    """
    tols = sorted({r["tolerance_ms"] for r in rows})
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    metrics = ["rpeak_sensitivity", "rpeak_precision", "rpeak_f1"]
    x = np.arange(len(tols))
    width = 0.2
    colors = ["#D68910", "#1B3A6B", "#8E44AD"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [_fmean([r[m] for r in rows if r["tolerance_ms"] == tl
                        and np.isfinite(r[m])]) for tl in tols]
        ax.bar(x + (i - 1) * width, vals, width, label=m.replace("_", " "),
               color=c, edgecolor="k", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"±{t:.0f} ms" for t in tols])
    ax.set_ylabel("Mean R-peak metric (one-to-one)")
    ax.set_ylim(0, 1)
    ax.set_title(title or "R-peak sensitivity / precision / F1 at "
                          "different match tolerances")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_report_waveforms(clean, noisy, methods, fs, path,
                          achieved=None, title=None):
    """Stacked waveform panel for the report: clean, noisy, denoised
    (bandpass), and the error waveform (estimate - clean)."""
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    t = np.arange(clean.size) / fs

    est = np.asarray(methods.get("bandpass_0.5_40hz", noisy), dtype=float)
    err = est - clean

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    n_rows = 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 1.4 * n_rows),
                             sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(t, clean, color="#1B3A6B", lw=0.9)
    axes[0].set_title(f"Clean reference ({clean.size / fs:.1f} s segment)"
                      + (f", mean {clean.mean():+.4f} mV" if achieved is None
                         else ""))
    axes[0].set_ylabel("mV")
    axes[1].plot(t, noisy, color="#C0392B", lw=0.9)
    axes[1].set_title(f"Noisy input (mixer SNR {achieved:.2f} dB)"
                      if achieved is not None else "Noisy input")
    axes[1].set_ylabel("mV")
    axes[2].plot(t, est, color="#27AE60", lw=0.9)
    axes[2].set_title("Denoised (bandpass 0.5-40 Hz) — "
                      f"RMSE vs clean = {_rmse(clean, est):.5f} mV")
    axes[2].set_ylabel("mV")
    axes[3].plot(t, err, color="#D68910", lw=0.7, alpha=0.85)
    axes[3].axhline(0, color="k", lw=0.4)
    axes[3].set_title(f"Error waveform (est - clean); "
                      f"RMSE = {_rmse(clean, est):.5f} mV, "
                      f"mean = {err.mean():+.5f} mV")
    axes[3].set_ylabel("error (mV)")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title or "Report: clean vs noisy vs denoised (bandpass) + error")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_report_rpeak(clean, est, fs, path, method="bandpass_0.5_40hz",
                      ref_peaks=None, det_peaks=None, title=None):
    """Waveform panel with R-peak overlays: clean signal, reference R-peaks
    (expert `.atr` or automated detector) and detected peaks of ``est``."""
    clean = np.asarray(clean, dtype=float)
    est = np.asarray(est, dtype=float)
    t = np.arange(clean.size) / fs

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    ax.plot(t, clean, color="#1B3A6B", lw=0.9, label="Clean")
    if ref_peaks is not None and len(ref_peaks):
        ax.plot(ref_peaks / fs, clean[ref_peaks], "v", color="#C0392B",
                ms=8, label=f"Reference R-peaks (n={len(ref_peaks)})")
    if det_peaks is not None and len(det_peaks):
        ax.plot(det_peaks / fs, est[det_peaks], "^", color="#27AE60",
                ms=6, label=f"Detected peaks of {method} (n={len(det_peaks)})")
    ax.set_title(title or "R-peak overlay: reference vs detected")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_method_bars(rows, metric, snr_level, path, title=None,
                     noise_key="noise_category"):
    """Horizontal bar chart of one metric's mean value per method at a
    fixed target SNR, one group per noise type."""
    subset = [r for r in rows if r["target_snr_db"] == snr_level]
    if not subset:
        raise ValueError(f"no rows at target SNR {snr_level}")
    by_noise = _group_rows(subset, noise_key)
    methods = []
    for row in subset:
        if row["method"] not in methods:
            methods.append(row["method"])

    n = len(by_noise)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 0.5 * len(methods) + 1.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    y = np.arange(len(methods))
    for ax, (noise_name, noise_rows) in zip(axes, sorted(by_noise.items())):
        means = []
        for method in methods:
            vals = [r[metric] for r in noise_rows if r["method"] == method
                    and np.isfinite(r.get(metric, float("nan")))]
            means.append(float(np.mean(vals)) if vals else np.nan)
        colors = [_method_color(m) for m in methods]
        ax.barh(y, means, color=colors)
        ax.set_title(f"{str(noise_name).replace('_', ' ')} @ {snr_level} dB")
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([m.replace("_", " ") for m in methods])
    axes[0].set_xlabel(metric)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
