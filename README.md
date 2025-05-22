# Real-Time Wavefunction Collapse Simulation

This repository contains the full implementation of the real-time collapse model described in the paper:

**"Real-Time Wavefunction Collapse from Free Energy Minimization: A Deterministic Variational Approach"**  
Stelios Savva (2025)

---

## 🧠 Overview

The simulation evolves a quantum wavefunction \( \psi(x,t) \) under a nonlinear, non-Hermitian evolution equation derived from a variational free energy principle:

\[
i\hbar \frac{d\psi}{dt} = H_0 \psi + iT(\ln |\psi|^2 + 1)\psi
\]

This equation governs deterministic wavefunction collapse via an entropy-energy competition mechanism, without invoking measurement, noise, or decoherence.

---

## 📂 Repository Contents

- `real_simulation.py` – Main Python script for running the simulation
- `real_time_config.json` – Configurable parameters (domain size, time step, temperature, etc.)
- `README.md` – This documentation file

---

## ⚙️ How It Works

The code:
- Initializes a spatial grid and a Gaussian wavefunction
- Evolves the wavefunction using spectral methods (FFT) for the kinetic term
- Applies a nonlinear entropy gradient term based on the Shannon entropy functional
- Records diagnostic quantities: entropy, energy, norm, expectation values, snapshots
- Outputs results in `.npy` and `.csv` formats for post-processing and figure generation

---

## 🔍 Key Features

- Fully deterministic, real-time collapse model
- No projection postulates or stochastic noise
- High-resolution spatial and temporal evolution
- Output includes:
  - `entropy_history.npy`, `energy_history.npy`
  - `free_energy_history.npy`, `snapshots.npy`
  - `energy_dissipation_rate.csv`

---

## 📊 How to Run

1. Make sure you have Python 3.7+ and `numpy`, `scipy`, `matplotlib`, and `pandas`.
2. Edit `real_time_config.json` to customize simulation parameters.
3. Run the simulation:

```bash
python real_simulation.py --config_file real_time_config.json

