
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

| MATLAB concept | Step 1 implementation |
| --- | --- |
| App Designer window and controls | `BasicWindow` and standard PyQt6 widgets |
| Button-pushed callback | `run_simulation` method connected to the run button |
| Simulation parameters | `SimulationParameters` dataclass |
| MATLAB numeric arrays and operations | NumPy arrays and the shared `simulation.py` implementation |
| Status/progress output | Progress bar, read-only text area, and message boxes |

The interface and numerical code are deliberately kept separate. This makes
the algorithm easier to test and lets later interface steps reuse the same
simulation engine.

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