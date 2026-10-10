"""Threat-detection metrics computed on the characterization symbol."""

import numpy as np


def cp_correlation(signal, cp_len, fft_size):
    """Correlate the received CP with the tail of the useful symbol."""
    cp = signal[:, :cp_len]
    useful = signal[:, cp_len:cp_len + fft_size]
    tail = useful[:, -cp_len:]

    numerator = np.sum(cp * np.conj(tail), axis=1)
    denominator = np.sqrt(
        np.sum(np.abs(cp) ** 2, axis=1) * np.sum(np.abs(tail) ** 2, axis=1)
    ) + 1e-12
    return np.abs(numerator / denominator)


def spectral_concentration(freq, used_subcarrier_indices):
    """Fraction of energy that lies in the used subcarriers."""
    used = freq[:, used_subcarrier_indices]
    total = np.sum(np.abs(freq) ** 2, axis=1)
    used_energy = np.sum(np.abs(used) ** 2, axis=1)
    return used_energy / (total + 1e-12)