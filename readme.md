
## Goal

Translate the MATLAB App Designer simulation into a small Python application
that can configure and run the OFDM simulation. This is the first functional
port, not a recreation of the complete MATLAB interface.

## What this step contains

- A minimal PyQt6 window with controls for antennas, channel taps, cyclic
  prefix, FFT size, OFDM symbols, trial count, jammer power, and modulation.
- Validation before the simulation starts.
- A progress bar and text output for simulation messages and key detection
  results.
- The Python simulation and parameter definitions imported from the project
  root; the numerical core is not duplicated in this step folder.

## MATLAB-to-Python mapping

| MATLAB concept | Step 1 implementation | Exact Python location |
| --- | --- | --- |
| App Designer window and controls | `BasicWindow` and standard PyQt6 widgets | `conversion_progressions/step 1/main.py` → `BasicWindow.__init__` and `BasicWindow._spin` |
| Button-pushed callback | `run_simulation` method connected to the run button | `conversion_progressions/step 1/main.py` → `BasicWindow.run_simulation` |
| Simulation parameters | `SimulationParameters` dataclass | `parameters.py` → `SimulationParameters` |
| Parameter validation | GUI-level validation before starting the run | `parameters.py` → `validate_parameters` |
| MATLAB numeric arrays and operations | NumPy arrays and the shared simulation core | `simulation.py` → `run_simulation` and helper functions such as `_generate_symbol_signals` |
| Status/progress output | Progress bar, read-only text area, and message boxes | `conversion_progressions/step 1/main.py` → `BasicWindow._update_progress` and `BasicWindow.run_simulation` |

The interface and numerical code are deliberately kept separate. This makes
the algorithm easier to test and lets later interface steps reuse the same
simulation engine.

## How the MATLAB main simulation becomes Python

The MATLAB simulation is embedded in the App Designer callback
`StartSimulationButtonPushed`. Step 1 does not copy that large callback into
the GUI. Instead, it separates the callback's responsibilities:

1. `BasicWindow.run_simulation` reads the values from the PyQt6 controls and
   places them in a `SimulationParameters` object. MATLAB's local variables
   such as `num_receive_antennas` and `snr_dB_list` become named fields on that
   object. The SNR sweep is built with `numpy.arange(-5.0, 31.0, 1.0)`, which
   includes -5 through 30 dB.
2. `validate_parameters` checks the settings before starting any simulation.
   If a setting is invalid, the GUI displays a warning and returns, rather
   than entering the MATLAB nested loops with incompatible array dimensions.
3. The callback prepares the progress bar and output area, then calls the
   shared `simulation.run_simulation(params, ...)` function. That function is
   the Python counterpart to the numerical body of the MATLAB callback.
4. The simulation returns a result dictionary containing BER arrays, detection
   metrics, the seed, and other data. Step 1 displays a short summary and the
   compliant and CP-violating cyclic-prefix correlation values. Other Python
   components can use the same result dictionary for plots and detailed
   dashboards.
5. A `finally` block restores the cursor and re-enables the run button whether
   the run succeeds or fails. Errors are presented in a critical message box.

At a high level, the MATLAB calculation maps to the shared Python simulation
like this:

| MATLAB main-callback section | Python implementation | Exact file and function |
| --- | --- | --- |
| Set antenna, channel, CP, FFT, modulation, trial, and SNR parameters | UI controls, `SimulationParameters`, and `config.py` defaults | `conversion_progressions/step 1/main.py` → `BasicWindow.run_simulation`; `parameters.py` → `SimulationParameters`; `config.py` → default settings |
| `switch` to define modulation symbols, normalize them, create bit mappings | Constellation and bit mapping utilities | `ofdm_utils.py` → `create_constellation`, `create_bit_mapping` |
| Allocate arrays for bits, received signals, BER, and jammer statistics | NumPy arrays allocated inside the simulation core | `simulation.py` → `run_simulation` |
| Monte-Carlo trial loop and random channel generation | Trial loop using a NumPy RNG | `simulation.py` → `run_simulation` (`for trial in range(params.num_trials)`) |
| Loop over SNR and OFDM symbols | Python loops over SNR points and symbol indices | `simulation.py` → `run_simulation` (`for snr_index, snr_db in enumerate(params.snr_dB_list)`) |
| Generate user and jammer signals, add CP, convolve with channels | Signal generation helper | `simulation.py` → `_generate_symbol_signals`, `_add_cp`, `_complex_normal` |
| Truncate CP, transform to frequency domain, select active tones | FFT and active-subcarrier selection | `simulation.py` → `_channel_to_frequency`; `ofdm_utils.py` → `create_used_subcarrier_indices` |
| SVD jammer estimates and projection matrices | Dominant-vector and matrix-solver helpers | `simulation.py` → `_dominant_vector`, `_lstsq_or_zeros` |
| Zero-forcing/equalization, nearest constellation decode, and error counts | Least-squares equalization and nearest-neighbor decoding | `simulation.py` → `run_simulation`; `ofdm_utils.py` → `minimum_distance_decode` |
| Normalize BER, calculate summary metrics, export tables | Final metrics and CSV export | `simulation.py` → `run_simulation` (final BER normalization and export block) |
| Update UI plots and progress during the run | Progress callback and log callback | `conversion_progressions/step 1/main.py` → `BasicWindow._update_progress`; `simulation.py` → `run_simulation` callback parameters |

The nested-loop structure is preserved conceptually: each trial creates a
channel realization, the SNR loop evaluates noise levels, and the OFDM-symbol
loop creates and receives symbols. NumPy operates on whole arrays where
practical, while loops remain where per-tone SVD and detection are required.
The Python code uses zero-based indexing, so MATLAB index expressions must be
adjusted when translating slices and subcarrier positions. For example,
MATLAB's one-based `cyclic_prefix_length+1` start becomes Python's
zero-based slice start `cp`.

The Step 1 UI's `main` block at the bottom of `main.py` is separate from the
MATLAB numerical main callback: it creates `QApplication`, constructs and
shows `BasicWindow`, and starts Qt's event loop. This replaces launching the
MATLAB app itself. The simulation algorithm is delegated to
`simulation.run_simulation`, keeping application startup, user interaction,
and numerical processing as distinct responsibilities.

The MATLAB operator walkthrough (`TVWSOperatorButtonPushed`) is not part of
the Step 1 basic conversion. Step 1 focuses on running and summarizing the
numerical OFDM simulation; richer threat/scenario workflows belong to the
later application stages.

## Current limitations

- The simulation runs synchronously on the GUI thread. The window may stop
  responding while a long run is in progress.
- Results are shown as text; there is no full set of MATLAB-style plots or
  experiment-history workflow yet.
- This step uses the Python implementation in the project root. It is not a
  line-by-line translation of the MATLAB `uifigure` component construction.

## Run

From the project root, with the dependencies in `requirements.txt` installed:

```powershell
python "conversion_progressions/step 1/main.py"
```

## Completion check

1. Launch the window and confirm all simulation inputs are available.
2. Run with valid parameters and confirm progress and completion output appear.
3. Enter an invalid parameter combination and confirm validation prevents the
   run and displays the reason.
