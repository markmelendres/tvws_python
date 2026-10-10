"""OFDM building blocks: subcarriers, constellations, bit mapping, decoding."""

import numpy as np


def create_used_subcarrier_indices(num_subcarriers):
    """Return the MATLAB-compatible used-subcarrier indices.

    The MATLAB reference adds 32 to signed offsets for its 1-based indexing.
    Convert that mapping to NumPy's 0-based indexing before wrapping bins.
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
    """Construct the complex modulation constellation for the chosen scheme."""
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
    """Create the bit-to-symbol mapping for a constellation with the given size."""
    bits_per_symbol = int(np.log2(num_points))
    mappings = np.zeros((num_points, bits_per_symbol), dtype=np.int8)
    for index in range(num_points):
        for bit in range(bits_per_symbol):
            mappings[index, bit] = (index >> (bits_per_symbol - 1 - bit)) & 1
    return bits_per_symbol, mappings


def minimum_distance_decode(symbols, constellation, bit_mapping):
    """Decode received symbols by nearest-neighbor matching against the constellation."""
    symbols = np.asarray(symbols).reshape(-1)
    distances = np.abs(symbols[:, None] - constellation[None, :]) ** 2
    indices = np.argmin(distances, axis=1)
    return indices, bit_mapping[indices]