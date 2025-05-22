import numpy as np
from numpy.fft import fft, ifft

def compute_kinetic(psi, k):
    return -0.5 * ifft(k**2 * fft(psi))

def compute_entropy(psi, x):
    rho = np.abs(psi)**2
    log_rho = np.log(np.maximum(rho, 1e-10))  # regularization
    S = -np.trapezoid(rho * log_rho, x)
    return S, rho

def compute_mu(psi, kinetic, ent_force, x):
    integrand = np.real(np.conjugate(psi) * (kinetic + ent_force))
    return np.trapezoid(integrand, x)
