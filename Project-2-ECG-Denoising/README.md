# Project 2 — ECG Denoising Pipeline Comparison

## Introduction

This project compares digital signal-processing methods for reducing common noise sources in electrocardiogram (ECG) signals. A synthetic clean ECG is generated, noise is added, and each denoising method is evaluated against the known clean reference.

## Data and noise model

The experiment uses a synthetic, normal-beat ECG at 250 Hz. The noisy signal combines:
- Baseline wander
- 50 Hz powerline interference
- Broadband EMG-like noise

Synthetic data provides a known clean reference for objective SNR and RMSE calculations. It does not establish performance on clinical ECG recordings.

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib

## Denoising methods

1. Median-filter baseline removal
2. Butterworth bandpass filtering (0.5–40 Hz)
3. IIR notch filtering (50 Hz)
4. FFT-domain soft-thresholding
5. Combined median baseline correction, notch filtering, and bandpass filtering

## Evaluation

Each method is compared with the clean synthetic reference using:
- Signal-to-Noise Ratio (SNR)
- SNR improvement relative to the noisy input
- Root Mean Square Error (RMSE)

The script also creates waveform comparison and SNR-improvement plots.

## How to run

From the repository root:

```bash
cd Project-2-ECG-Denoising
pip install -r requirements.txt
cd src
python run_comparison.py
```

## Output

The run creates an `outputs/` directory inside the project folder containing:
- `metrics.json`
- `denoising_comparison.png`
- `snr_improvement.png`

## What I learned

- Common ECG noise sources and their effects
- Median, Butterworth, and notch filtering
- Frequency-domain thresholding
- Quantitative signal-quality evaluation using SNR and RMSE
- Comparing independent filters with a combined processing pipeline

## Future improvements

- Validate the methods using real ECG recordings.
- Explore wavelet-based denoising.
- Compare filter parameters across different noise levels.
- Assess preservation of clinically relevant ECG morphology.

## References

- [SciPy signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [PhysioNet](https://physionet.org/)

## Disclaimer

This is an educational experiment using synthetic ECG data. It is not intended for clinical diagnosis or medical decision-making.
