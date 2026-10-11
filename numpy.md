# NumPy Function Reference — TVWS OFDM Simulator

This document explains every NumPy function, method, and operator used in the three numerical modules:

| File | Role |
| --- | --- |
| `simulation.py` | Monte-Carlo engine: waveform generation, channel, jammer, projection, BER |
| `ofdm_utils.py` | Subcarrier selection, constellations, bit mapping, nearest-neighbor decoding |
| `metrics.py` | CP correlation and spectral concentration detector metrics |

Conventions used below: `n_rx` = receive antennas, `n_tx` = transmit antennas, `n_fft` = FFT size, `cp` = cyclic-prefix length, `taps` = channel taps, `n_used` = number of active subcarriers.

---

## 1. Quick index

| Category | Functions |
| --- | --- |
| Random numbers | `np.random.default_rng`, `np.random.SeedSequence`, `rng.standard_normal`, `rng.integers`, `rng.random` |
| Array creation | `np.zeros`, `np.zeros_like`, `np.ones`, `np.full`, `np.eye`, `np.array`, `np.arange` |
| Array joining / reshaping | `np.concatenate`, `.reshape`, `.T`, `.copy`, `.astype`, `np.asarray`, slicing, `[:, None]` |
| Math | `np.sqrt`, `np.abs`, `np.conj`, `np.log2`, `np.mean`, `np.std`, `np.sum`, `np.sort`, `np.argmin`, `np.nan` |
| Signal processing | `np.fft.fft`, `np.fft.ifft`, `np.convolve` |
| Linear algebra | `np.linalg.svd`, `np.linalg.lstsq`, `np.linalg.matrix_rank`, `np.outer`, `np.tensordot`, `@` |

---

## 2. Random number generation

### `np.random.default_rng(seed)`
Creates a modern `Generator` object (PCG64 algorithm). All randomness in the simulation (bits, channels, noise, jammer) comes from this single object, so the same `seed` reproduces the exact same run.

- **Used in:** `run_simulation` → `rng = np.random.default_rng(seed)`

### `np.random.SeedSequence().entropy`
`SeedSequence()` gathers fresh entropy from the operating system; `.entropy` exposes that large integer. It is used as an automatic seed when the caller passes `seed=None`, and the value is logged and returned so a run can be reproduced later.

- **Used in:** `run_simulation` → `seed = np.random.SeedSequence().entropy`

### `rng.standard_normal(shape)`
Returns samples from a Gaussian with mean 0 and variance 1. `shape` can be an int or tuple.

- **Used in:** `_complex_normal`

```python
np.sqrt(0.5) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
```

Real and imaginary parts each have variance 0.5, so the complex value has total variance 1 (circularly symmetric complex Gaussian, CN(0,1)). This models noise, the jammer, and Rayleigh channel taps.

### `rng.integers(low, high, size, dtype)`
Draws random integers in `[low, high)`.

- **Used in:** generating all transmitted bits at once

```python
bits = rng.integers(0, 2, size=(num_trials, num_ofdm_symbols, n_tx, bits_per_symbol, n_used), dtype=np.int8)
```

`dtype=np.int8` stores each bit in 1 byte instead of 8, which matters because this array is large.

### `rng.random()`
Returns one float uniformly distributed in `[0, 1)`.

- **Used in:** random jammer power — `10 ** (jammer_power_dB * rng.random() / 10)` picks a random jammer power between 0 dB and the maximum for each trial.

---

## 3. Array creation

### `np.zeros(shape, dtype)`
Allocates an array filled with zeros. Pre-allocating is faster than growing arrays inside loops.

- **Used in:** `symbols_freq` (`dtype=complex`), `legit` (received buffer), `ber_*` accumulators, `received_by_tone` buffers, `jammer_compliant_frequency`, `spatial_interference`, `create_bit_mapping`'s lookup table, and the rank-deficient fallback in `_lstsq_or_zeros`.

### `np.zeros_like(a)`
Zeros with the same shape and dtype as `a`.

- **Used in:** `comp = np.zeros_like(legit)`, `noncomp = np.zeros_like(legit)`, and `jammer_noncompliant_frequency = np.zeros_like(jammer_compliant_frequency)`.

### `np.ones(n)`
Array of ones. Used for the fallback dominant vector when SVD returns no singular values:

```python
np.ones(n_rx, dtype=complex) / np.sqrt(n_rx)   # unit-norm uniform vector
```

### `np.full(shape, fill_value)`
Array filled with a constant. Used with `np.nan` to build live BER arrays where SNR points not yet simulated appear as gaps in the plot.

```python
"BER_jammerless": np.full(snr_count, np.nan)
```

### `np.nan`
The floating-point "not a number" constant. Plotting libraries skip NaN points.

### `np.eye(n)`
Identity matrix of size `n × n`.

- **Used in:** building projection matrices `P = I − v vᴴ`, the "no projection" case for the legitimate signal, and the `n_rx == 1` stable fallback.

### `np.array(list, dtype)`
Converts a Python list into an ndarray.

- **Used in:** `create_constellation` (BPSK/QPSK/16QAM/64QAM point lists, `dtype=complex`).

### `np.arange(start, stop, step)`
Evenly spaced integers, `stop` excluded.

- **In `ofdm_utils.create_used_subcarrier_indices`:** builds ranges of signed subcarrier offsets such as `np.arange(-26, -21)` → `[-26, -25, -24, -23, -22]`.
- **In `_map_symbols`:** `np.arange(bps - 1, -1, -1)` produces `[bps-1, …, 1, 0]`, the shifts for the MSB-first bit weights.

---

## 4. Joining, reshaping, copying, indexing

### `np.concatenate(arrays, axis)`
Joins arrays along an existing axis.

- **`_add_cp`:** `np.concatenate((x[..., -cp_len:], x), axis=-1)` prepends the last `cp_len` samples to the symbol — this *is* the cyclic prefix. `axis=-1` means "last axis," so it works for 1-D and multi-antenna 2-D arrays alike.
- **`create_used_subcarrier_indices`:** stitches the separate index ranges into one list (skipping the DC bin and pilot positions).

### `array.reshape(-1)` / `np.asarray(x).reshape(-1)`
Flattens to 1-D (`-1` = "infer this dimension").

- **Used in:** flattening decoded symbols and bits before decoding and before comparing with `true_bits`.

### `array.T`
Transpose (swaps axes for 2-D arrays).

- `y_legit_freq[:, used_idx].T` converts shape `(n_rx, n_used)` into `(n_used, n_rx)` so the first axis is "tone".
- `estimate.T.reshape(-1)` reorders the equalizer output `(n_tx, n_symbols)` into a symbol-major stream matching the bit layout.

### `array.copy()`
Creates an independent copy. Characterization arrays are copied so later loop iterations do not overwrite them, because slicing in NumPy returns *views*, not copies.

### `array.astype(dtype)`
Converts dtype.

- `bits_slice.astype(int)` — avoids int8 overflow during bit-weight math.
- `fft_indices.astype(int)` — makes the indices usable for array indexing.

### `np.asarray(x)`
Converts input to an ndarray without copying if it already is one. Used in `minimum_distance_decode` and for `b`'s shape in `_lstsq_or_zeros`.

### Slicing and broadcasting syntax
| Expression | Meaning |
| --- | --- |
| `x[..., -cp_len:]` | Last `cp_len` samples along the last axis, any number of leading axes |
| `legit[:, start:stop]` | All antennas, samples `cp` to `cp + n_fft` — **removes the cyclic prefix** |
| `symbols_freq[:, used_idx] = …` | Assign to only the active subcarriers (fancy indexing) |
| `symbols[:, None]` | Adds a new axis: shape `(N,)` → `(N, 1)` for broadcasting |
| `constellation[None, :]` | Shape `(M,)` → `(1, M)` |
| `spatial_interference /= column_sum[None, :]` | Divides each column by its own sum |
| `column_sum[column_sum == 0] = 1.0` | Boolean-mask assignment, avoids divide-by-zero |

---

## 5. Elementwise math and statistics

### `np.sqrt(x)`
Square root, used for scaling:

- `np.sqrt(0.5)` in the complex Gaussian
- `np.sqrt(n_fft)` for FFT/IFFT scaling so power is preserved between domains
- `np.sqrt(jammer_power_linear)` and `np.sqrt(noise_variance)` to turn power into amplitude
- constellation normalization in `create_constellation`
- the denominator of `cp_correlation`

### `np.abs(x)`
Magnitude. For complex numbers returns `√(re² + im²)`.

- Energy: `np.abs(x) ** 2`
- `np.abs(vec_n) ** 2` — per-antenna power of the dominant interference vector
- Distance in the decoder, and final correlation value in `cp_correlation`

### `np.conj(x)`
Complex conjugate (flips the sign of the imaginary part).

- `np.outer(vec, np.conj(vec))` creates `v vᴴ`
- `cp * np.conj(tail)` in `cp_correlation` — the standard complex correlation

### `np.log2(x)`
Base-2 logarithm. `int(np.log2(num_points))` converts constellation size to bits per symbol (4 points → 2 bits, 64 points → 6 bits).

### `np.mean(x, axis=None)`
Arithmetic mean.

- `np.mean(np.abs(constellation) ** 2)` — average symbol energy for normalization
- `np.mean(cp_correlation(...))` — average over receive antennas
- `np.mean(spatial_interference, axis=1)` — average across subcarriers
- Averaging detector metrics across trials

### `np.std(x, axis=None)`
Standard deviation. `np.std(spatial_interference, axis=1)` gives the spread of per-antenna interference across subcarriers.

### `np.sum(x, axis=None)`
Sum of elements. `axis` selects which dimension to collapse.

| Use | Purpose |
| --- | --- |
| `np.sum(true_bits != decoded)` | Counts bit errors (booleans sum as 0/1) |
| `np.sum(np.abs(snapshot) ** 2)` | Total jammer energy before/after projection |
| `np.sum(spatial_interference, axis=0)` | Per-subcarrier normalization constant |
| `np.sum(cp * np.conj(tail), axis=1)` | Correlation numerator per antenna |
| `np.sum(np.abs(freq) ** 2, axis=1)` | Total energy per antenna (spectral concentration) |

### `np.sort(a)`
Returns a sorted copy. Used so active subcarrier indices are ascending after the wrap-around modulo.

### `np.argmin(a, axis)`
Index of the smallest value. In `minimum_distance_decode`, `np.argmin(distances, axis=1)` returns, for each received symbol, the index of the closest constellation point.

### Operators
- `10 ** (x / 10)` — dB to linear conversion.
- `(a + n_fft // 2 - 1) % n_fft` — modulo wraps negative frequencies into 0…`n_fft-1`.
- `1 << np.arange(...)` — bit shift, creates powers of two (`[8, 4, 2, 1]` for 4 bits).
- `!=`, `==` — elementwise comparison producing boolean arrays.

---

## 6. Signal-processing functions

### `np.fft.fft(x, n=None, axis=-1)`
Discrete Fourier transform (time → frequency). `n` zero-pads or truncates to that length.

- **`_channel_to_frequency`:** `np.fft.fft(channel_time, n=n_fft, axis=-1) / np.sqrt(n_fft)` zero-pads the short channel impulse response to `n_fft` points to obtain the per-subcarrier channel gain `H[k]`.
- **`to_freq` (receiver):** `np.fft.fft(x, axis=1) / np.sqrt(n_fft)` demodulates each antenna's received CP-stripped symbol. Dividing by `√N` keeps the transform unitary.

### `np.fft.ifft(x, axis=-1)`
Inverse DFT (frequency → time).

- Used on the jammer's frequency-domain noise and on the user's subcarrier symbols (`axis=1` for the antenna dimension). Multiplying by `√n_fft` offsets NumPy's `1/N` normalization so average power is preserved.

### `np.convolve(a, b)`
Full 1-D linear convolution; output length is `len(a) + len(b) − 1`.

- Models the multipath channel: `np.convolve(user_channel_time[rx, tx], symbols_time[tx])`. Output length is `n_fft + cp + taps − 1`, which is exactly the `rx_len` computed in the code.
- `np.convolve` only accepts 1-D inputs, so the code loops over antennas.

---

## 7. Linear algebra

### `np.linalg.svd(A, full_matrices=False)`
Singular value decomposition `A = U Σ Vᴴ`. Returns `U`, singular values `S` (descending), and `Vᴴ`. `full_matrices=False` returns the compact form.

- **`_dominant_vector`:** `U[:, 0]` is the strongest spatial direction across the receive antennas (the dominant jammer direction); `S[0]` is its strength.

### `np.outer(a, b)`
Outer product → matrix `a bᵀ`. With `np.conj`, `np.outer(v, np.conj(v))` is the rank-1 projector onto `v`. The code uses it as:

```python
P = np.eye(n_rx) - np.outer(vec, np.conj(vec))
```

`P` removes the dominant interference direction (null-space projection, the core jammer-mitigation step).

### `@` (matrix multiplication)
- `pn @ jammer_noncomp_snapshot` — jammer after projection
- `projector @ channel` and `projector @ received` — projected channel and observation in the equalizer

### `np.linalg.matrix_rank(A)`
Number of linearly independent rows/columns (via SVD with a tolerance). In `_lstsq_or_zeros`, rank `0` means the matrix is effectively all zeros, so the solve is skipped and zeros are returned.

### `np.linalg.lstsq(A, b, rcond=None)`
Solves `min ‖A x − b‖²`. `[0]` selects the solution `x`. This acts as the **zero-forcing / least-squares equalizer** recovering transmitted symbols from received samples. `rcond=None` uses the machine-precision default cutoff for small singular values.

### `np.tensordot(a, b, axes)`
Generalized dot product over chosen axes.

- **`_map_symbols`:** `np.tensordot(bits_slice.astype(int), weights, axes=([1], [0]))` contracts the bits-per-symbol axis with the weights `[2^(b-1), …, 2, 1]`, converting each group of bits (MSB first) into an integer index. Input `(n_tx, bits_per_symbol, n_used)` → output `(n_tx, n_used)`.

---

## 8. How the functions combine, by project function

### `simulation.py`

| Function | NumPy calls |
| --- | --- |
| `_complex_normal` | `np.sqrt`, `rng.standard_normal` |
| `_channel_to_frequency` | `np.fft.fft`, `np.sqrt` |
| `_add_cp` | `np.concatenate` |
| `_map_symbols` | `np.arange`, `np.tensordot`, `.astype`, bit shift |
| `_generate_symbol_signals` | `np.fft.ifft`, `np.zeros`, `np.zeros_like`, `np.convolve`, `np.sqrt`, `_complex_normal` |
| `_lstsq_or_zeros` | `np.linalg.matrix_rank`, `np.linalg.lstsq`, `np.zeros`, `np.asarray` |
| `_dominant_vector` | `np.linalg.svd`, `np.ones`, `np.sqrt` |
| `_should_emit` | (none — uses `time.perf_counter`) |
| `run_simulation` | `default_rng`, `SeedSequence`, `rng.integers`, `rng.random`, `np.zeros`, `np.full`, `np.nan`, `np.eye`, `np.outer`, `np.conj`, `np.abs`, `np.sum`, `np.mean`, `np.std`, `np.fft.fft`, `.copy`, `.T`, `.reshape`, `@` |

### `ofdm_utils.py`

| Function | NumPy calls | Result |
| --- | --- | --- |
| `create_used_subcarrier_indices` | `np.concatenate`, `np.arange`, `%`, `.astype`, `np.sort`, `int()` | Sorted active FFT bins (52 for `n_fft = 64`) |
| `create_constellation` | `np.array`, `np.sqrt`, `np.mean`, `np.abs` | Unit-average-energy symbol set |
| `create_bit_mapping` | `np.log2`, `np.zeros`, shifts, `& 1` | `(num_points, bits_per_symbol)` bit lookup table |
| `minimum_distance_decode` | `np.asarray`, `.reshape`, broadcasting, `np.abs`, `np.argmin` | Symbol indices and decoded bits |

### `metrics.py`

#### `cp_correlation(signal, cp_len, fft_size)`

```python
cp     = signal[:, :cp_len]                    # first cp_len samples (the prefix)
useful = signal[:, cp_len:cp_len + fft_size]   # the FFT window
tail   = useful[:, -cp_len:]                   # last cp_len samples of the symbol
```

| NumPy call | Role |
| --- | --- |
| Slicing | Isolates CP and symbol tail per antenna |
| `np.conj(tail)` | Conjugate for complex correlation |
| `np.sum(cp * np.conj(tail), axis=1)` | Numerator: `Σ cp[n]·tail*[n]` per antenna |
| `np.abs(cp) ** 2`, `np.sum(..., axis=1)` | Energy of CP and tail |
| `np.sqrt(E_cp * E_tail) + 1e-12` | Normalizer (the epsilon prevents division by zero) |
| `np.abs(numerator / denominator)` | Normalized correlation magnitude |

By the Cauchy–Schwarz inequality the result lies in `[0, 1]`. A legitimate OFDM signal gives a value near 1 because the CP is a copy of the symbol tail; noise-like or CP-violating jammers give values near 0.

#### `spectral_concentration(freq, used_subcarrier_indices)`

| NumPy call | Role |
| --- | --- |
| `freq[:, used_subcarrier_indices]` | Fancy indexing: keeps only active bins |
| `np.sum(np.abs(freq) ** 2, axis=1)` | Total energy per antenna |
| `np.sum(np.abs(used) ** 2, axis=1)` | Energy in the active band per antenna |
| `used_energy / (total + 1e-12)` | Fraction of energy inside the allocated band, in `[0, 1]` |

A clean OFDM signal puts most of its energy on active tones; a broadband jammer spreads energy over unused bins, lowering the ratio.

---

## 9. Shape walk-through of one OFDM symbol

```text
bits[trial, symbol]              (n_tx, bits_per_symbol, n_used)
  └─ _map_symbols (tensordot) →  (n_tx, n_used)            constellation indices → complex symbols
  └─ symbols_freq[:, used_idx]   (n_tx, n_fft)             placed on active bins, zeros elsewhere
  └─ ifft + _add_cp →            (n_tx, n_fft + cp)        time domain with prefix
  └─ convolve with channel →     (n_rx, n_fft + cp + taps − 1)
  └─ [:, start:stop] + noise →   (n_rx, n_fft)             prefix removed
  └─ fft / √n_fft →              (n_rx, n_fft)             back to frequency domain
  └─ [:, used_idx].T →           (n_used, n_rx)            per-tone snapshot
  └─ lstsq equalizer →           (n_tx, n_symbols)         symbol estimates
  └─ argmin decode →             bits, compared with true_bits for BER
```

---

## 10. Review notes

1. **Inconsistent CP-correlation inputs in the live update.** The characterization metrics call `cp_correlation(rx_*_char, cp, n_fft)` on the *full* received signal (CP included). The live metrics call it on `legit[:, start:stop]`, which has *already removed* the CP. In that case `signal[:, :cp_len]` is the first part of the useful symbol rather than the prefix, so the live values are not comparable with the final ones. Passing `legit`, `total_compliant`, and `total_noncompliant` (without the slice) would make them consistent.
2. **`_lstsq_or_zeros` only catches the all-zero case.** `matrix_rank(A) > 0` is false only when `A` is (numerically) zero. Rank-deficient but non-zero matrices still go to `lstsq`, which handles them with a minimum-norm solution, so the docstring slightly overstates the fallback.
3. **Shared `bit_mapping` is unused by `_map_symbols`.** Bits become constellation indices through MSB-first weights, which matches `create_bit_mapping`'s ordering. Both agree today, but if either one changes, the transmit and receive mappings will silently diverge.
