
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParameters:
    num_receive_antennas: int
    num_transmit_antennas: int
    num_channel_taps: int
    cyclic_prefix_length: int
    num_subcarriers: int
    num_ofdm_symbols: int
    jammer_power_dB: float
    modulation_scheme: str
    random_jammer_power_flag: bool
    num_trials: int
    snr_dB_list: np.ndarray


def validate_parameters(params):
    if params.num_receive_antennas < 1:
        raise ValueError("Receive antennas must be >= 1.")
    if params.num_transmit_antennas < 1:
        raise ValueError("Transmit antennas must be >= 1.")
    if params.num_channel_taps < 1:
        raise ValueError("Channel taps must be >= 1.")
    if params.num_subcarriers < 8:
        raise ValueError("FFT size must be >= 8.")
    if params.cyclic_prefix_length < 0:
        raise ValueError("Cyclic prefix length cannot be negative.")
    if params.cyclic_prefix_length >= params.num_subcarriers:
        raise ValueError("Cyclic prefix length must be smaller than FFT size.")
    if params.num_channel_taps > params.cyclic_prefix_length:
        raise ValueError(
            "For the intended CP-OFDM model, channel taps "
            "should not exceed the cyclic prefix length."
        )
    if params.num_ofdm_symbols < 1:
        raise ValueError("Number of OFDM symbols must be >= 1.")
    if params.num_trials < 1:
        raise ValueError("Number of trials must be >= 1.")