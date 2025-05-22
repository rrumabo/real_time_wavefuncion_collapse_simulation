import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d
from utils import compute_entropy, compute_kinetic, compute_mu

def run_simulation(config):
    L = config['L']
    N = config['N']
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    k = fftfreq(N, d=dx) * 2 * np.pi

    x0 = config.get('x0', -5.0)
    sigma0 = config.get('sigma0', 1.0)
    p0 = config.get('p0', 0.0)
    normalization = (1 / (sigma0 * np.sqrt(np.pi)))**0.5
    psi = normalization * np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * p0 * x)

    psi_final, results = evolve_wavefunction(psi, config, x, k)
    return x, psi_final, results

def evolve_wavefunction(psi, config, x, k):
    num_steps = config['num_steps']
    d_tau = config['d_tau']
    T_param = config['T_param']
    alpha = config['alpha']
    imaginary_time = config.get('imaginary_time', True)
    hbar = config.get('hbar', 1.0)
    m = config.get('m', 1.0)
    snapshot_interval = config.get('snapshot_interval', 200)

    norm_history, entropy_history, energy_history = [], [], []
    free_energy_history, x_mean_history, x2_mean_history, snapshots = [], [], [], []

    for n in range(num_steps):
        S, rho = compute_entropy(psi, x)
        kinetic = compute_kinetic(psi, k)
        log_rho = np.log(np.maximum(rho, 1e-12))
        ent_force = alpha * T_param * (log_rho + 1) * psi

        if imaginary_time:
            mu = compute_mu(psi, kinetic, ent_force, x)
            psi = psi - d_tau * (kinetic + ent_force - mu * psi)
            psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
        else:
            psi = psi + (-1j / hbar) * d_tau * (kinetic + ent_force)

        norm_val = np.trapezoid(np.abs(psi)**2, x)
        norm_history.append(norm_val)
        S, rho = compute_entropy(psi, x)
        entropy_history.append(S)

        psi_k = fft(psi)
        kinetic_energy = np.trapezoid(np.abs(psi_k)**2 * (k**2) / (2 * m), k)
        V_eff = alpha * T_param * (log_rho + 1)
        potential_energy = np.trapezoid(rho * V_eff, x)
        total_energy = kinetic_energy + potential_energy
        energy_history.append(total_energy)

        F = total_energy - T_param * S
        free_energy_history.append(F)

        x_mean = np.trapezoid(x * rho, x)
        x2_mean = np.trapezoid(x**2 * rho, x)
        x_mean_history.append(x_mean)
        x2_mean_history.append(x2_mean)

        if n % snapshot_interval == 0:
            snapshots.append(psi.copy())

    dE_dtau = gaussian_filter1d(np.gradient(energy_history, d_tau), sigma=2)
    dS_dtau = gaussian_filter1d(np.gradient(entropy_history, d_tau), sigma=2)
    threshold = 1e-8
    T_computed = np.where(np.abs(dS_dtau) > threshold, dE_dtau / dS_dtau, np.nan)

    return psi, {
        'tau': np.arange(num_steps) * d_tau,
        'norm_history': np.array(norm_history),
        'entropy_history': np.array(entropy_history),
        'energy_history': np.array(energy_history),
        'free_energy_history': np.array(free_energy_history),
        'T_computed': np.array(T_computed),
        'x_mean_history': np.array(x_mean_history),
        'x2_mean_history': np.array(x2_mean_history),
        'snapshots': np.array(snapshots)
    }
