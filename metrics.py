"""Metrics that measure how strongly a received waveform matches the expected
OFDM structure.

The characterization symbol is used as a compact fingerprint of the signal. The
functions in this module compare the cyclic prefix with the tail of the useful
symbol and estimate how much of the signal energy remains inside the active
subcarriers. Those numbers feed the detector logic that decides whether a signal
is safe or should be treated as a threat.
"""

import numpy as np


def cp_correlation(signal, cp_len, fft_size):
    """Measure how well the cyclic prefix matches the end of the active OFDM symbol.

    A legitimate OFDM waveform should preserve a strong phase and amplitude
    relationship between the CP and the symbol tail. When the correlation drops,
    the receiver suspects that the signal structure has been distorted by a
    jammer or CP-violating waveform. The returned value is normalized and stays
    in the range [0, 1], which makes it convenient for the UI and detector
    thresholds.
    """
    cp = signal[:, :cp_len]
    useful = signal[:, cp_len:cp_len + fft_size]
    tail = useful[:, -cp_len:]

    numerator = np.sum(cp * np.conj(tail), axis=1)
    denominator = np.sqrt(
        np.sum(np.abs(cp) ** 2, axis=1) * np.sum(np.abs(tail) ** 2, axis=1)
    ) + 1e-12
    return np.abs(numerator / denominator)


def spectral_concentration(freq, used_subcarrier_indices):
    """Return the fraction of total signal energy contained in active subcarriers.

    This metric helps separate clean OFDM traffic from broadband interference.
    A valid signal concentrates most of its power in the allocated bins, while a
    jammer or misaligned waveform spreads power across a wider spectrum. The
    result is a ratio between 0 and 1 that is used to compare signal quality
    across different reception conditions.
    """
    used = freq[:, used_subcarrier_indices]
    total = np.sum(np.abs(freq) ** 2, axis=1)
    used_energy = np.sum(np.abs(used) ** 2, axis=1)
    return used_energy / (total + 1e-12)
