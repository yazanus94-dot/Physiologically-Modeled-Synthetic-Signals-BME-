# Makefile for the biomedical signal-processing portfolio.
#
# Usage:
#   make test           # run all three projects' unit test suites
#   make synthetic      # run all three synthetic experiments
#   make real           # run all three real-PhysioNet experiments (download if needed)

PY ?= python

.PHONY: test synthetic real rpeaks mitbih bidmc

test:
	$(PY) -m pytest project-1-ecg-arrhythmia/tests project-2-ecg-denoising/tests project-3-ppg-hrv/tests benchmark-tasks/mitbih_nstdb_denoising_benchmark/tests

# Synthetic experiments (no PhysioNet download required).
synthetic:
	cd project-1-ecg-arrhythmia/src && $(PY) run_pipeline.py --mode synthetic && cd ../..
	cd project-2-ecg-denoising/src && $(PY) run_comparison.py && cd ../..
	cd project-3-ppg-hrv/src && $(PY) run_pipeline.py && cd ../..

# Real-PhysioNet experiments.
real: mitbih bidmc
	$(MAKE) real-nstdb

mitbih:
	cd project-1-ecg-arrhythmia/src && $(PY) run_pipeline.py --mode mitbih --protocol classic && cd ../..
	cd project-1-ecg-arrhythmia/src && $(PY) run_pipeline.py --mode mitbih --protocol leakage_audited && cd ../..
	cd project-1-ecg-arrhythmia/src && $(PY) run_pipeline.py --mode rpeaks && cd ../..

real-nstdb:
	cd project-2-ecg-denoising/src && $(PY) run_real_experiment.py --download && cd ../..

bidmc:
	cd project-3-ppg-hrv/src && $(PY) run_bidmc_experiment.py --download && cd ../..
