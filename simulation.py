
"""Monte-Carlo TVWS OFDM simulation core (no GUI dependencies)."""

import time

import numpy as np
import pandas as pd

from config import (
    BER_EXPORT_FILE,
    PROJECTION_EXPORT_FILE,
)
from metrics import cp_correlation, spectral_concentration
from ofdm_utils import (
    create_bit_mapping,
    create_constellation,
    create_used_subcarrier_indices,
    minimum_distance_decode,
)
from parameters import validate_parameters


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _complex_normal(rng, shape):
    """Unit-variance circular complex Gaussian (real drawn first, then imag)."""
    return np.sqrt(0.5) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def _channel_to_frequency(channel_time, n_fft):
    return np.fft.fft(channel_time, n=n_fft, axis=-1) / np.sqrt(n_fft)


def _add_cp(x, cp_len):
    """Prepend a cyclic prefix along the last axis."""
    if cp_len == 0:
        return x
    return np.concatenate((x[..., -cp_len:], x), axis=-1)


def _map_symbols(bits_slice, constellation):
    """bits_slice: [Tx, bits_per_symbol, tones] -> constellation points [Tx, tones]."""
    bps = bits_slice.shape[1]
    weights = 1 << np.arange(bps - 1, -1, -1)
    idx = np.tensordot(bits_slice.astype(int), weights, axes=([1], [0]))
    return constellation[idx]


def _generate_symbol_signals(rng, params, bits_trial_symbol, constellation,
                             used_idx, user_channel_time, jammer_channel_time,
                             jammer_power_linear):
    """Build one OFDM symbol and the signals received by every antenna.

    Returns (legitimate_rx, compliant_rx, noncompliant_rx,
             symbols_time, jammer_time, jammer_time_noncompliant).
    """
    n_fft = params.num_subcarriers
    cp = params.cyclic_prefix_length
    n_rx = params.num_receive_antennas
    n_tx = params.num_transmit_antennas

    # OFDM-compliant jammer
    jammer_freq = _complex_normal(rng, n_fft)
    jammer_time = _add_cp(np.sqrt(n_fft) * np.fft.ifft(jammer_freq), cp)
    jammer_time *= np.sqrt(jammer_power_linear)

    # CP-violating jammer
    jammer_time_nc = _complex_normal(rng, n_fft + cp)
    jammer_time_nc *= np.sqrt(jammer_power_linear)

    # Legitimate user
    symbols_used = _map_symbols(bits_trial_symbol, constellation)
    symbols_freq = np.zeros((n_tx, n_fft), dtype=complex)
    symbols_freq[:, used_idx] = symbols_used
    symbols_time = _add_cp(np.sqrt(n_fft) * np.fft.ifft(symbols_freq, axis=1), cp)

    rx_len = n_fft + cp + params.num_channel_taps - 1
    legit = np.zeros((n_rx, rx_len), dtype=complex)
    comp = np.zeros_like(legit)
    noncomp = np.zeros_like(legit)

    for rx in range(n_rx):
        for tx in range(n_tx):
            legit[rx] += np.convolve(user_channel_time[rx, tx], symbols_time[tx])
        comp[rx] = np.convolve(jammer_channel_time[rx], jammer_time)
        noncomp[rx] = np.convolve(jammer_channel_time[rx], jammer_time_nc)

    return legit, comp, noncomp, symbols_time, jammer_time, jammer_time_nc


def _lstsq_or_zeros(A, b, n_tx):
    if np.linalg.matrix_rank(A) > 0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    output_shape = (n_tx,) + np.asarray(b).shape[1:]
    return np.zeros(output_shape, dtype=complex)


def _dominant_vector(snapshot, n_rx):
    U, S, _ = np.linalg.svd(snapshot, full_matrices=False)
    if len(S) > 0:
        return U[:, 0], S
    return np.ones(n_rx, dtype=complex) / np.sqrt(n_rx), S


def _should_emit(last_emit_time, min_interval_seconds=0.12, force=False):
    if force:
        return True
    return (time.perf_counter() - last_emit_time) >= min_interval_seconds


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def run_simulation(params, progress_callback=None, message_callback=None,
                   stop_callback=None, live_callback=None, seed=None):

    validate_parameters(params)

    if seed is None:
        seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)

    n_fft = params.num_subcarriers
    cp = params.cyclic_prefix_length
    n_rx = params.num_receive_antennas
    n_tx = params.num_transmit_antennas
    taps = params.num_channel_taps

    used_idx = create_used_subcarrier_indices(n_fft)
    n_used = len(used_idx)
    constellation = create_constellation(params.modulation_scheme)
    bits_per_symbol, bit_mapping = create_bit_mapping(len(constellation))

    def log(message):
        if message_callback is not None:
            message_callback(str(message))

    log("")
    log("=" * 64)
    log("TVWS OFDM SIMULATION")
    log("=" * 64)
    log(f"Random seed      : {seed}")
    log(f"Receive antennas : {n_rx}")
    log(f"Transmit antennas: {n_tx}")
    log(f"Channel taps     : {taps}")
    log(f"FFT size         : {n_fft}")
    log(f"CP length        : {cp}")
    log(f"Used subcarriers : {n_used}")
    log(f"OFDM symbols     : {params.num_ofdm_symbols}")
    log(f"Trials           : {params.num_trials}")
    log(f"Jammer power     : {params.jammer_power_dB:.1f} dB")
    log(f"Modulation       : {params.modulation_scheme}")
    log(f"Bits/symbol      : {bits_per_symbol}")

    bits = rng.integers(
        0, 2,
        size=(params.num_trials, params.num_ofdm_symbols, n_tx,
              bits_per_symbol, n_used),
        dtype=np.int8,
    )

    snr_count = len(params.snr_dB_list)
    ber_jammerless = np.zeros(snr_count)
    ber_compliant = np.zeros(snr_count)
    ber_noncompliant = np.zeros(snr_count)
    ber_projections = np.zeros((taps, snr_count))

    characterization_symbol = params.num_ofdm_symbols - 1
    characterization_trial = -1

    rx_legit_char = rx_comp_char = rx_noncomp_char = None
    time_legit_char = time_comp_char = time_noncomp_char = None
    legit_char_freq = comp_char_freq = noncomp_char_freq = None
    comp_jammer_char_freq = noncomp_jammer_char_freq = None
    live_threat_detection = None
    trial_threat_detections = []

    jammer_compliant_frequency = np.zeros(
        (params.num_trials, n_used, n_rx, params.num_ofdm_symbols), dtype=complex
    )
    jammer_noncompliant_frequency = np.zeros_like(jammer_compliant_frequency)
    spatial_interference = np.zeros((n_rx, n_used))
    jammer_energy_before = 0.0
    jammer_energy_after = 0.0

    start_time = time.perf_counter()
    total_trial_work = params.num_trials * snr_count * params.num_ofdm_symbols
    completed_work = 0
    last_progress_emit = 0.0
    last_live_emit = 0.0
    last_mitigation_emit = 0.0

    start = cp
    stop = cp + n_fft

    # ========================================================
    # TRIAL LOOP
    # ========================================================
    for trial in range(params.num_trials):

        if stop_callback is not None and stop_callback():
            return None

        jammer_channel_time = _complex_normal(rng, (n_rx, taps))
        user_channel_time = _complex_normal(rng, (n_rx, n_tx, taps))
        user_channel_freq = _channel_to_frequency(user_channel_time, n_fft)
        # (jammer_channel_freq was computed but never used in the original)

        if params.random_jammer_power_flag:
            jammer_power_linear = n_tx * 10 ** (
                params.jammer_power_dB * rng.random() / 10)
        else:
            jammer_power_linear = n_tx * 10 ** (params.jammer_power_dB / 10)

        # ----------------------------------------------------
        # Detection / BER pass
        # ----------------------------------------------------
        for snr_index, snr_db in enumerate(params.snr_dB_list):

            noise_variance = taps * n_tx * 10 ** (-snr_db / 10)
            received_by_tone = {
                "legitimate": np.zeros(
                    (n_used, n_rx, params.num_ofdm_symbols), dtype=complex),
                "compliant": np.zeros(
                    (n_used, n_rx, params.num_ofdm_symbols), dtype=complex),
                "noncompliant": np.zeros(
                    (n_used, n_rx, params.num_ofdm_symbols), dtype=complex),
                "jammer_compliant": np.zeros(
                    (n_used, n_rx, params.num_ofdm_symbols), dtype=complex),
                "jammer_noncompliant": np.zeros(
                    (n_used, n_rx, params.num_ofdm_symbols), dtype=complex),
            }

            for symbol_index in range(params.num_ofdm_symbols):

                (legit, comp, noncomp, s_time, j_time, j_time_nc) = (
                    _generate_symbol_signals(
                        rng, params, bits[trial, symbol_index], constellation,
                        used_idx, user_channel_time, jammer_channel_time,
                        jammer_power_linear)
                )

                total_compliant = legit + comp
                total_noncompliant = legit + noncomp

                noise = _complex_normal(rng, (n_rx, n_fft)) * np.sqrt(noise_variance)

                y_legit_time = legit[:, start:stop] + noise
                y_comp_time = total_compliant[:, start:stop] + noise
                y_noncomp_time = total_noncompliant[:, start:stop] + noise
                jam_comp_time = comp[:, start:stop]
                jam_noncomp_time = noncomp[:, start:stop]

                def to_freq(x):
                    return np.fft.fft(x, axis=1) / np.sqrt(n_fft)

                y_legit_freq = to_freq(y_legit_time)
                y_comp_freq = to_freq(y_comp_time)
                y_noncomp_freq = to_freq(y_noncomp_time)
                jam_comp_freq = to_freq(jam_comp_time)
                jam_noncomp_freq = to_freq(jam_noncomp_time)

                if (snr_index == snr_count - 1
                        and symbol_index == characterization_symbol):
                    characterization_trial = trial
                    rx_legit_char = legit.copy()
                    rx_comp_char = total_compliant.copy()
                    rx_noncomp_char = total_noncompliant.copy()
                    time_legit_char = s_time.copy()
                    time_comp_char = j_time.copy()
                    time_noncomp_char = j_time_nc.copy()
                    legit_char_freq = y_legit_freq.copy()
                    comp_char_freq = y_comp_freq.copy()
                    noncomp_char_freq = y_noncomp_freq.copy()
                    comp_jammer_char_freq = jam_comp_freq.copy()
                    noncomp_jammer_char_freq = jam_noncomp_freq.copy()
                    live_threat_detection = {
                        "CP correlation - legitimate": float(
                            np.mean(cp_correlation(rx_legit_char, cp, n_fft))),
                        "CP correlation - OFDM compliant": float(
                            np.mean(cp_correlation(rx_comp_char, cp, n_fft))),
                        "CP correlation - CP violating": float(
                            np.mean(cp_correlation(rx_noncomp_char, cp, n_fft))),
                        "Used-band concentration - legitimate": float(
                            np.mean(spectral_concentration(
                                legit_char_freq, used_idx))),
                        "Used-band concentration - OFDM compliant": float(
                            np.mean(spectral_concentration(
                                comp_char_freq, used_idx))),
                        "Used-band concentration - CP violating": float(
                            np.mean(spectral_concentration(
                                noncomp_char_freq, used_idx))),
                    }
                    trial_threat_detections.append(live_threat_detection)

                received_by_tone["legitimate"][:, :, symbol_index] = (
                    y_legit_freq[:, used_idx].T)
                received_by_tone["compliant"][:, :, symbol_index] = (
                    y_comp_freq[:, used_idx].T)
                received_by_tone["noncompliant"][:, :, symbol_index] = (
                    y_noncomp_freq[:, used_idx].T)
                received_by_tone["jammer_compliant"][:, :, symbol_index] = (
                    jam_comp_freq[:, used_idx].T)
                received_by_tone["jammer_noncompliant"][:, :, symbol_index] = (
                    jam_noncomp_freq[:, used_idx].T)

                if (live_callback is not None
                        and _should_emit(last_live_emit, min_interval_seconds=0.2)):
                    live_detection = {
                        "CP correlation - legitimate": float(
                            np.mean(cp_correlation(legit[:, start:stop], cp, n_fft))),
                        "CP correlation - OFDM compliant": float(
                            np.mean(cp_correlation(total_compliant[:, start:stop], cp, n_fft))),
                        "CP correlation - CP violating": float(
                            np.mean(cp_correlation(total_noncompliant[:, start:stop], cp, n_fft))),
                        "Used-band concentration - legitimate": float(
                            np.mean(spectral_concentration(y_legit_freq, used_idx))),
                        "Used-band concentration - OFDM compliant": float(
                            np.mean(spectral_concentration(y_comp_freq, used_idx))),
                        "Used-band concentration - CP violating": float(
                            np.mean(spectral_concentration(y_noncomp_freq, used_idx))),
                    }
                    live_update = {
                        "kind": "received_signals",
                        "legitimate_frequency": y_legit_freq,
                        "compliant_frequency": y_comp_freq,
                        "noncompliant_frequency": y_noncomp_freq,
                        "used_subcarrier_indices": used_idx,
                        "characterization_trial": trial,
                        "characterization_trial_count": params.num_trials,
                        "characterization_symbol": symbol_index,
                        "cyclic_prefix_length": cp,
                        "threat_detection": live_detection,
                    }
                    live_callback(live_update)
                    last_live_emit = time.perf_counter()

                completed_work += 1
                should_emit_progress = (
                    progress_callback is not None and (
                        completed_work == total_trial_work
                        or _should_emit(last_progress_emit, min_interval_seconds=0.15)
                    )
                )
                if should_emit_progress:
                    progress_callback(completed_work, total_trial_work)
                    last_progress_emit = time.perf_counter()
                if stop_callback is not None and stop_callback():
                    return None

            for tone_index, subcarrier in enumerate(used_idx):
                jammer_comp_snapshot = received_by_tone[
                    "jammer_compliant"][tone_index]
                jammer_noncomp_snapshot = received_by_tone[
                    "jammer_noncompliant"][tone_index]
                jammer_compliant_frequency[trial, tone_index] = (
                    jammer_comp_snapshot)
                jammer_noncompliant_frequency[trial, tone_index] = (
                    jammer_noncomp_snapshot)

                vec_c, _ = _dominant_vector(jammer_comp_snapshot, n_rx)
                vec_n, singular_values = _dominant_vector(
                    jammer_noncomp_snapshot, n_rx)
                pc = np.eye(n_rx) - np.outer(vec_c, np.conj(vec_c))
                pn = np.eye(n_rx) - np.outer(vec_n, np.conj(vec_n))
                if n_rx == 1:
                    # MATLAB's scalar projection is applied to both sides of
                    # the least-squares solve, so its nonzero round-off value
                    # cancels. Use the equivalent stable identity projection.
                    pc_for_ber = np.eye(n_rx)
                    pn_for_ber = np.eye(n_rx)
                else:
                    pc_for_ber = pc
                    pn_for_ber = pn
                jammer_energy_before += float(
                    np.sum(np.abs(jammer_noncomp_snapshot) ** 2))
                projected_jammer = pn @ jammer_noncomp_snapshot
                jammer_energy_after += float(np.sum(np.abs(projected_jammer) ** 2))
                is_last_mitigation_sample = (
                    trial == params.num_trials - 1
                    and tone_index == n_used - 1
                )
                if (
                    live_callback is not None
                    and _should_emit(
                        last_mitigation_emit,
                        min_interval_seconds=0.2,
                        force=is_last_mitigation_sample,
                    )
                ):
                    energy_removed_percent = 100 * (
                        1 - jammer_energy_after / jammer_energy_before
                    )
                    live_callback({
                        "kind": "live_mitigation",
                        "energy_removed_percent": float(energy_removed_percent),
                        "trial": trial,
                        "trial_count": params.num_trials,
                    })
                    last_mitigation_emit = time.perf_counter()
                if singular_values.size:
                    spatial_interference[:, tone_index] += (
                        np.abs(vec_n) ** 2 * singular_values[0])

                channel = user_channel_freq[:, :, subcarrier]
                signal_sets = (
                    ("legitimate", received_by_tone["legitimate"][tone_index],
                     np.eye(n_rx), 1.0),
                    ("compliant", received_by_tone["compliant"][tone_index],
                     pc_for_ber, np.sqrt(n_fft)),
                    ("noncompliant", received_by_tone["noncompliant"][tone_index],
                     pn_for_ber, np.sqrt(n_fft)),
                )
                decoded = {}
                for name, received, projector, channel_scale in signal_sets:
                    estimate = _lstsq_or_zeros(
                        channel_scale * projector @ channel,
                        projector @ received,
                        n_tx,
                    )
                    _, decoded[name] = minimum_distance_decode(
                        estimate.T.reshape(-1), constellation, bit_mapping)

                true_bits = bits[trial, :, :, :, tone_index].reshape(-1)
                ber_jammerless[snr_index] += np.sum(
                    true_bits != decoded["legitimate"].reshape(-1))
                ber_compliant[snr_index] += np.sum(
                    true_bits != decoded["compliant"].reshape(-1))
                ber_noncompliant[snr_index] += np.sum(
                    true_bits != decoded["noncompliant"].reshape(-1))

                identity = np.eye(n_rx)
                for projection_index in range(taps):
                    estimate = _lstsq_or_zeros(
                        np.sqrt(n_fft) * identity @ channel,
                        identity @ received_by_tone["noncompliant"][tone_index],
                        n_tx,
                    )
                    _, decoded_projection = minimum_distance_decode(
                        estimate.T.reshape(-1), constellation, bit_mapping)
                    ber_projections[projection_index, snr_index] += np.sum(
                        true_bits != decoded_projection.reshape(-1))

            if live_callback is not None:
                processed_bits = (
                    (trial + 1) * n_tx * params.num_ofdm_symbols
                    * bits_per_symbol * n_used)
                live_ber = {
                    "kind": "live_ber",
                    "SNR_dB": params.snr_dB_list,
                    "BER_jammerless": np.full(snr_count, np.nan),
                    "BER_compliant": np.full(snr_count, np.nan),
                    "BER_noncompliant": np.full(snr_count, np.nan),
                }
                live_ber["BER_jammerless"][:snr_index + 1] = (
                    ber_jammerless[:snr_index + 1] / processed_bits)
                live_ber["BER_compliant"][:snr_index + 1] = (
                    ber_compliant[:snr_index + 1] / processed_bits)
                live_ber["BER_noncompliant"][:snr_index + 1] = (
                    ber_noncompliant[:snr_index + 1] / processed_bits)
                if _should_emit(last_live_emit, min_interval_seconds=0.25, force=(snr_index == snr_count - 1)):
                    live_callback(live_ber)
                    last_live_emit = time.perf_counter()

        if live_callback is not None:
            trial_update = {
                "kind": "received_signals",
                "legitimate_frequency": legit_char_freq,
                "compliant_frequency": comp_jammer_char_freq,
                "noncompliant_frequency": noncomp_jammer_char_freq,
                "used_subcarrier_indices": used_idx,
                "characterization_trial": trial,
                "characterization_trial_count": params.num_trials,
                "characterization_symbol": characterization_symbol,
                "cyclic_prefix_length": cp,
                "threat_detection": live_threat_detection,
                "jammer_compliant_freq_domain": jammer_compliant_frequency[:trial + 1],
                "jammer_noncompliant_freq_domain": jammer_noncompliant_frequency[:trial + 1],
                "spatial_interference_distribution": spatial_interference.copy(),
            }
            if _should_emit(last_live_emit, min_interval_seconds=0.3, force=(trial == params.num_trials - 1)):
                live_callback(trial_update)
                last_live_emit = time.perf_counter()

        log(f"Trial {trial + 1}/{params.num_trials} complete.")

    # ========================================================
    # Normalize BER
    # ========================================================
    total_bits = (params.num_trials * n_tx * params.num_ofdm_symbols
                  * bits_per_symbol * n_used)
    ber_jammerless /= total_bits
    ber_compliant /= total_bits
    ber_noncompliant /= total_bits
    ber_projections /= total_bits

    # ========================================================
    # Spatial normalization
    # ========================================================
    column_sum = np.sum(spatial_interference, axis=0)
    column_sum[column_sum == 0] = 1.0
    spatial_interference /= column_sum[None, :]
    spatial_average = np.mean(spatial_interference, axis=1)
    spatial_std = np.std(spatial_interference, axis=1)

    # ========================================================
    # Received-signal characterization
    # ========================================================
    if rx_legit_char is None:
        raise RuntimeError(
            "Unable to construct requested trial/symbol characterization.")
    if (legit_char_freq is None or comp_char_freq is None
            or noncomp_char_freq is None or live_threat_detection is None):
        raise RuntimeError("Received-signal characterization is incomplete.")
    if comp_jammer_char_freq is None or noncomp_jammer_char_freq is None:
        raise RuntimeError("Jammer characterization signals are incomplete.")

    # ========================================================
    # Threat detection / mitigation metrics
    # ========================================================
    threat_detection = {
        name: float(np.mean([trial_metrics[name]
                             for trial_metrics in trial_threat_detections]))
        for name in live_threat_detection
    }

    if jammer_energy_before == 0:
        raise RuntimeError("No jammer energy was captured for mitigation.")
    jammer_before = jammer_energy_before
    jammer_after = jammer_energy_after

    mitigation_metrics = {
        "Jammer energy before projection": float(jammer_before),
        "Jammer energy after projection": float(jammer_after),
        "Jammer energy removed (%)": float(
            100 * (1 - jammer_after / jammer_before)),
        "Projection dimensions": taps,
        "Receive spatial dimensions": n_rx,
        "Mitigation status": "Projection processing completed",
    }

    # ========================================================
    # Export tables
    # ========================================================
    ber_table = pd.DataFrame({
        "SNR_dB": params.snr_dB_list,
        "BER_legit": ber_jammerless,
        "BER_compliant": ber_compliant,
        "BER_noncompliant": ber_noncompliant,
    })

    projection_data = {"SNR_dB": params.snr_dB_list}
    for index in range(taps):
        projection_data[f"BER_Null{index + 1}"] = ber_projections[index]
    projection_table = pd.DataFrame(projection_data)

    ber_table.to_csv(BER_EXPORT_FILE, index=False)
    projection_table.to_csv(PROJECTION_EXPORT_FILE, index=False)

    log("")
    log(f"BER exported to {BER_EXPORT_FILE}")
    log(f"Projection BER exported to {PROJECTION_EXPORT_FILE}")
    log("")
    log("=" * 64)
    log("SIMULATION COMPLETE")
    log("=" * 64)
    log(f"Elapsed time: {(time.perf_counter() - start_time) / 60:.2f} minutes")

    return {
        "seed": int(seed),
        "SNR_dB": params.snr_dB_list,
        "BER_jammerless": ber_jammerless,
        "BER_compliant": ber_compliant,
        "BER_noncompliant": ber_noncompliant,
        "BER_projections": ber_projections,
        "used_subcarrier_indices": used_idx,
        "spatial_interference_distribution": spatial_interference,
        "spatial_interference_distribution_sc_avg": spatial_average,
        "spatial_interference_distribution_sc_std": spatial_std,
        "jammer_compliant_freq_domain": jammer_compliant_frequency,
        "jammer_noncompliant_freq_domain": jammer_noncompliant_frequency,
        "characterization_trial": characterization_trial,
        "characterization_trial_count": params.num_trials,
        "characterization_symbol": characterization_symbol,
        "received_legitimate": rx_legit_char,
        "received_compliant": rx_comp_char,
        "received_noncompliant": rx_noncomp_char,
        "legitimate_frequency": legit_char_freq,
        "compliant_frequency": comp_char_freq,
        "noncompliant_frequency": noncomp_char_freq,
        "compliant_jammer_frequency": comp_jammer_char_freq,
        "noncompliant_jammer_frequency": noncomp_jammer_char_freq,
        "legitimate_time": time_legit_char,
        "compliant_time": time_comp_char,
        "noncompliant_time": time_noncomp_char,
        "threat_detection": threat_detection,
        "mitigation_metrics": mitigation_metrics,
        "parameters": params,
    }