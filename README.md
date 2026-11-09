# Physiologically Modeled Synthetic Signals — Biomedical Engineering Portfolio

## Introduction

This repository presents three educational projects combining biomedical engineering, Python programming, physiological signal modeling, digital signal processing, and machine learning.

The projects explore synthetic ECG and PPG signals. Synthetic-data results are useful for controlled experiments, but do not establish clinical performance or generalization to patient recordings.

## Projects

### 1. ECG Arrhythmia Beat Classification

A synthetic ECG classification exercise covering Pan–Tompkins-style R-peak detection, beat-level feature extraction, and Random Forest classification of Normal, PVC, and Supraventricular/Other beats.

[Read Project 1](Project-1-ECG-Arrhythmia/README.md)

### 2. ECG Denoising Pipeline Comparison

A comparison of median-filter baseline correction, Butterworth bandpass filtering, IIR notch filtering, FFT soft-thresholding, and a combined denoising pipeline. Performance is assessed against a known clean synthetic ECG using SNR, SNR improvement, and RMSE.

[Read Project 2](Project-2-ECG-Denoising/README.md)

### 3. PPG-Based Heart Rate Variability Analysis

A synthetic PPG exercise covering pulse generation, systolic peak detection, pulse-interval analysis, and time- and frequency-domain HRV metrics. The documented example also discusses how motion artifacts can affect HRV estimates.

[Read Project 3](Project-3-PPG-HRV/README.md)

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- WFDB (for optional real ECG record support described in Project 1)

## Development Workflow

1. Model or generate a physiological waveform.
2. Introduce or inspect noise and artifacts.
3. Preprocess the signal using digital filters.
4. Detect ECG R-peaks or PPG systolic peaks.
5. Extract features, calculate HRV metrics, or classify beats.
6. Compare results with synthetic ground truth and visualize performance.

## Learning Outcomes

- ECG and PPG waveform characteristics
- Physiological signal modeling
- Digital filtering and denoising
- QRS and systolic peak detection
- Feature extraction and machine learning classification
- Time-domain and frequency-domain HRV
- Quantitative evaluation using sensitivity, precision, SNR, and RMSE

## Future Improvements

- Test against independent real-world ECG and PPG recordings.
- Add reproducible dataset-loading and preprocessing workflows.
- Compare additional denoising and peak-detection methods.
- Evaluate classification with patient-independent data splits.
- Add automated tests, experiment configurations, and result reports.

## Running the Available Code

The ECG denoising source files and dependencies are included in this repository:

```bash
cd Project-2-ECG-Denoising
pip install -r requirements.txt
cd src
python run_comparison.py
```

The ECG arrhythmia and PPG/HRV READMEs describe their respective pipelines, but their Python source files and project-specific dependency files are not currently included in this repository. They are documented for reference and are not yet independently runnable from this repository.

## References

- [NumPy documentation](https://numpy.org/doc/stable/)
- [SciPy signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Matplotlib documentation](https://matplotlib.org/stable/)
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [WFDB Python documentation](https://wfdb.readthedocs.io/en/latest/)
- [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
- [Pan and Tompkins, 1985](https://doi.org/10.1109/TBME.1985.325532)
- [Task Force HRV standards, 1996](https://doi.org/10.1161/01.CIR.93.5.1043)

## About

**Yazan Aljabari**  
Biomedical Engineering | Information Technology | Biomedical Signal Processing | AI

[GitHub profile](https://github.com/yazanus94-dot)

## Disclaimer

These projects are for educational and research demonstration purposes. Synthetic signals and experimental algorithms are not medical devices and are not intended for clinical diagnosis or treatment.
