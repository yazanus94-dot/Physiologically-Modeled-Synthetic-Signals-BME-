# Physiologically Modeled Synthetic Signals | Biomedical Engineering Projects

## 👋 Introduction

Welcome to my **Biomedical Engineering and Signal Processing Portfolio**!

This repository presents practical projects focused on physiological signal generation, biomedical signal processing, ECG analysis, arrhythmia classification, signal denoising, and heart rate variability (HRV) estimation.

Using Python and scientific computing libraries, I explore how physiological signals can be modeled, processed, analyzed, and evaluated through signal-processing techniques and machine learning algorithms.

The portfolio currently covers three main areas: ECG Arrhythmia Classification, ECG Denoising, and PPG-Based Heart Rate Variability Analysis.

## 🎯 Project Objectives

- Explore physiological modeling of ECG and PPG signals.
- Generate synthetic biomedical signals with realistic characteristics and noise.
- Implement digital signal-processing techniques.
- Detect important physiological waveform features.
- Explore machine learning for ECG beat classification.
- Evaluate signal-processing algorithms using quantitative metrics.
- Understand the limitations of signal analysis in the presence of noise and artifacts.

## 🛠️ Technologies & Tools

| Technology | Application |
|---|---|
| Python | Programming and implementation |
| NumPy | Numerical computing and signal generation |
| SciPy | Digital filtering and signal processing |
| Matplotlib | Signal visualization and plotting |
| scikit-learn | Random Forest classification |
| WFDB | Optional support for working with real ECG records |

## 📂 Projects Overview

### 1. ECG Arrhythmia Beat Classification

**Techniques:** Pan-Tompkins-style R-peak Detection, Feature Extraction, Random Forest.

This project explores ECG beat classification using a physiologically informed synthetic ECG generator.

**Key Features:**
- Synthetic ECG generation at 250 Hz.
- Modeling of Normal (N), Premature Ventricular Contraction (PVC), and Supraventricular/Other (S) beats.
- R-peak detection using a Pan-Tompkins-style algorithm.
- Beat-level feature extraction using RR intervals and QRS morphology.
- Random Forest classification.
- Evaluation using sensitivity, precision, classification reports, and confusion matrices.

**Documented Results:**
- R-peak detection sensitivity: 92.0%.
- R-peak detection precision: 100%.
- Near-perfect classification on the documented held-out synthetic dataset.

**Important:** These results are from synthetic signals and should not be interpreted as clinical or real-world diagnostic performance.

### 2. ECG Denoising Pipeline Comparison

**Techniques:** Digital Filtering, Median Filtering, Notch Filtering, FFT-Based Denoising.

This project investigates different approaches to removing common types of noise from ECG signals.

**Simulated Noise Sources:**
- Baseline wander.
- Powerline interference (50/60 Hz).
- EMG and muscle-related noise.

**Denoising Methods:**
1. Median-filter baseline removal.
2. Butterworth bandpass filtering (0.5–40 Hz).
3. IIR notch filtering.
4. FFT-domain soft-thresholding.
5. Combined denoising pipeline.

**Evaluation Metrics:**
- Signal-to-Noise Ratio (SNR).
- SNR improvement.
- Root Mean Square Error (RMSE).

**Outputs:**
- ECG waveform comparisons.
- Denoising performance metrics.
- SNR improvement visualizations.
- JSON results.

### 3. PPG-Based Heart Rate Variability (HRV) Analysis

**Techniques:** PPG Signal Generation, Systolic Peak Detection, Time-Domain HRV, Frequency-Domain HRV.

This project focuses on generating synthetic photoplethysmography (PPG) signals and analyzing heart rate variability from detected pulse intervals.

**Key Features:**
- Synthetic PPG signal generation at 100 Hz.
- Physiologically motivated pulse morphology, including the systolic peak and dicrotic notch.
- Respiratory sinus arrhythmia (RSA) modeling.
- Simulated sensor noise and motion artifacts.
- Bandpass filtering and adaptive peak detection.
- Time-domain and frequency-domain HRV estimation.

**HRV Metrics:**
- Mean Heart Rate.
- SDNN.
- RMSSD.
- pNN50.
- VLF, LF, and HF spectral power.
- LF/HF ratio.

**Documented Results:**
- Peak detection sensitivity: 99.5%.
- Peak detection precision: 98.5%.
- Comparison between detected and ground-truth HRV metrics.

The analysis also demonstrates how motion artifacts can introduce errors into pulse interval measurements and inflate HRV metrics.

## ⚙️ Development Process

The general development workflow across these projects follows these stages:

1. **Physiological Modeling:** Define the signal characteristics and generate synthetic ECG or PPG waveforms.
2. **Noise Simulation:** Introduce different noise sources and artifacts to simulate measurement challenges.
3. **Signal Preprocessing:** Apply appropriate filtering and noise-reduction techniques.
4. **Feature Detection:** Identify important waveform characteristics, such as R-peaks and systolic peaks.
5. **Signal Analysis:** Extract features, calculate physiological metrics, or classify ECG beats.
6. **Evaluation:** Compare the processed results with known synthetic ground truth.
7. **Visualization:** Generate plots, metrics, and reports to understand algorithm performance.

## 📊 Evaluation & Performance

Each project uses evaluation methods appropriate to its objective.

| Project | Evaluation Approach |
|---|---|
| ECG Arrhythmia Classification | Sensitivity, Precision, Classification Report, Confusion Matrix |
| ECG Denoising | SNR, SNR Improvement, RMSE |
| PPG-Based HRV | Peak Detection Sensitivity, Precision, HRV Metric Comparison |

The use of synthetic signals allows controlled experiments because the generated signals have known reference characteristics. However, performance on synthetic data does not establish generalization to real clinical recordings.

## 🎓 What I Learned

Through these projects, I explored and developed practical knowledge of:

- Physiological signal modeling and waveform generation.
- ECG and PPG signal characteristics.
- Digital signal processing and filtering techniques.
- Pan-Tompkins-style QRS detection.
- Feature engineering and machine learning classification.
- Signal-to-noise ratio and error-based evaluation.
- Heart rate variability analysis.
- Time-domain and frequency-domain signal analysis.
- Signal visualization and interpretation.
- Limitations of algorithms when processing noisy physiological signals.

## 🚀 Future Improvements

Potential improvements and future development directions include:

- Integrating real ECG and PPG datasets from PhysioNet.
- Evaluating algorithms on independent and unseen recordings.
- Comparing additional machine learning classification algorithms.
- Exploring deep learning approaches for ECG classification.
- Improving noise removal and motion artifact rejection.
- Adding more robust peak detection methods.
- Evaluating performance across different signal qualities and sampling rates.
- Developing automated experiment reports and reproducible pipelines.
- Comparing synthetic and real physiological signals.

## 💻 How to Run the Projects

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- Basic knowledge of Python and signal processing.

### Step 1 – Clone the repository

```bash
git clone https://github.com/yazanus94-dot/Physiologically-Modeled-Synthetic-Signals-BME-.git
```

### Step 2 – Navigate to the repository

```bash
cd Physiologically-Modeled-Synthetic-Signals-BME-
```

### Step 3 – Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 4 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 – Run the ECG denoising comparison

For the ECG denoising scripts currently available at the repository root:

```bash
python run_comparison.py
```

The program generates comparison metrics and visualizations in the output directory.

**Note:** The ECG arrhythmia and PPG/HRV pipelines should be run from their respective project directories once their individual source files and dependency files are included in the repository. Follow the corresponding project README for their exact execution commands.

## 📁 Recommended Repository Structure

```text
Physiologically-Modeled-Synthetic-Signals-BME-/
│
├── README.md
├── Project-1-ECG-Arrhythmia/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   └── outputs/
│
├── Project-2-ECG-Denoising/
│   ├── README.md
│   ├── requirements.txt
│   ├── generate_noisy_signal.py
│   ├── denoise_methods.py
│   ├── run_comparison.py
│   └── outputs/
│
└── Project-3-PPG-HRV/
    ├── README.md
    ├── requirements.txt
    ├── src/
    └── outputs/
```

## 🎥 Project Demonstration

A video demonstration can be added here to showcase the physiological signal generation, processing pipelines, visualizations, and analysis results.

**Video:** 
https://github.com/user-attachments/assets/77b5a51c-06c4-42c3-877c-01ff2aac55a2

## 📚 References & Learning Resources

The following documentation and research resources provide background and technical guidance for the methods explored in this repository.

| Resource | Reference |
|---|---|
| NumPy | https://numpy.org/doc/stable/ |
| SciPy Signal Processing | https://docs.scipy.org/doc/scipy/reference/signal.html |
| Matplotlib | https://matplotlib.org/stable/ |
| scikit-learn Random Forest | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html |
| WFDB Python | https://wfdb.readthedocs.io/en/latest/ |
| PhysioNet MIT-BIH Arrhythmia Database | https://physionet.org/content/mitdb/ |
| Pan-Tompkins QRS Detection Paper | https://doi.org/10.1109/TBME.1985.325532 |
| Heart Rate Variability Standards | https://doi.org/10.1161/01.CIR.93.5.1043 |

## 👨‍💻 About Me

**Yazan Aljabari**

Biomedical Engineering | Information Technology | Biomedical Signal Processing | AI & Software Testing

I am a Biomedical Engineering graduate with an interest in combining engineering, programming, data analysis, and modern technologies to solve practical technical challenges.

My interests include biomedical signal processing, physiological modeling, medical data analysis, AI applications, and software quality assurance.

**GitHub:** https://github.com/yazanus94-dot

---

**Disclaimer:** This repository is intended for educational, research, and demonstration purposes. The synthetic signals and experimental algorithms are not medical devices and are not intended for clinical diagnosis or treatment.

*This portfolio is continuously evolving as I explore additional biomedical engineering methods, signal-processing techniques, and machine learning applications.*
