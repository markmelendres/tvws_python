"""Core OFDM primitives used by the simulator.

The functions in this module define how pilot and data subcarriers are selected,
how bit streams are mapped onto modulation symbols, and how noisy received
samples are decoded back into bits. These helpers keep the simulation logic
consistent with the MATLAB reference while still running efficiently in NumPy.
"""

import numpy as np


def create_used_subcarrier_indices(num_subcarriers):
    """Return the set of active OFDM subcarrier indices used by the MATLAB model.

    The reference implementation defines a centered pattern of used bins in
    signed frequency coordinates. This helper converts those signed offsets to
    the zero-based indexing scheme NumPy expects, wraps the values around the FFT
    circular bandwidth, and returns the bins in ascending order. The result is
    then reused across the transmit, receive, and distortion analysis stages.
    """
    if num_subcarriers < 8:
        raise ValueError("FFT size must be at least 8.")

    if num_subcarriers == 64:
        signed_indices = np.concatenate([
            np.arange(-26, -21),
            np.arange(-20, -7),
            np.arange(-6, 0),
            np.arange(1, 7),
            np.arange(8, 21),
            np.arange(22, 27),
        ])
    else:
        half = num_subcarriers // 2
        max_index = min(half - 1, int(26 * num_subcarriers / 64))
        signed_indices = np.concatenate([
            np.arange(-max_index, 0),
            np.arange(1, max_index + 1),
        ])

    fft_indices = (
        signed_indices + num_subcarriers // 2 - 1
    ) % num_subcarriers
    return np.sort(fft_indices.astype(int))


def create_constellation(modulation_scheme):
    """Build the normalized modulation constellation for a chosen scheme.

    Each supported scheme (BPSK, QPSK, 16QAM, 64QAM) defines a complex symbol
    set with the correct spacing and average energy. The returned values are
    scaled so their mean squared magnitude is normalized, which keeps the
    simulation signal level stable across different modulation orders.
    """
    if modulation_scheme == "BPSK":
        constellation = np.array([-1, 1], dtype=complex)
    elif modulation_scheme == "QPSK":
        constellation = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=complex)
    elif modulation_scheme == "16QAM":
        constellation = np.array([
            -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
            -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
             3 - 3j,  3 - 1j,  3 + 3j,  3 + 1j,
             1 - 3j,  1 - 1j,  1 + 3j,  1 + 1j,
        ], dtype=complex)
    elif modulation_scheme == "64QAM":
        levels = [-7, -5, -1, -3, 7, 5, 1, 3]
        constellation = np.array(
            [re + 1j * im for re in levels for im in levels], dtype=complex
        )
    else:
        raise ValueError(f"Unsupported modulation: {modulation_scheme}")

    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    return constellation


def create_bit_mapping(num_points):
    """Create a lookup table that converts bit groups into constellation indices.

    The mapping keeps the same bit ordering expected by the MATLAB reference:
    the most significant bit is read first for each symbol. This allows the
    transmitter to pack several bits into one QAM point and the receiver to
    recover them using the inverse scheme during demodulation.
    """
    bits_per_symbol = int(np.log2(num_points))
    mappings = np.zeros((num_points, bits_per_symbol), dtype=np.int8)
    for index in range(num_points):
        for bit in range(bits_per_symbol):
            mappings[index, bit] = (index >> (bits_per_symbol - 1 - bit)) & 1
    return bits_per_symbol, mappings


def minimum_distance_decode(symbols, constellation, bit_mapping):
    """Decode received complex values to bit groups using nearest-neighbor matching.

    Each received symbol is compared against the available constellation points,
    and the closest point is selected. The matching index is then mapped through
    the bit lookup table to recover the transmitted bits. This function is the
    demodulation counterpart of the mapper used during transmission.
    """
    symbols = np.asarray(symbols).reshape(-1)
    distances = np.abs(symbols[:, None] - constellation[None, :]) ** 2
    indices = np.argmin(distances, axis=1)
    return indices, bit_mapping[indices]
