# NumPy Function Reference and MATLAB Comparison: TVWS OFDM Simulator

This document explains every NumPy function, method, and operator used in the Python modules, shows the **MATLAB equivalent** from `Latest_SISO_APP_06152026.mlapp` (`StartSimulationButtonPushed`), and lists the **differences found** between the two implementations.

| Python file | Role | MATLAB source |
| --- | --- | --- |
| `simulation.py` | Monte-Carlo engine: waveforms, channel, jammer, projection, BER | Body of `StartSimulationButtonPushed` |
| `ofdm_utils.py` | Subcarrier selection, constellations, bit mapping, nearest-neighbor decoding | `switch (modulation_scheme)`, `de2bi`, `bi2de`, `min(abs(...).^2,[],2)` |
| `metrics.py` | CP correlation and spectral concentration | **No MATLAB counterpart** (new in Python) |

Conventions: `n_rx` = receive antennas, `n_tx` = transmit antennas, `n_fft` = FFT size, `cp` = cyclic-prefix length, `taps` = channel taps, `n_used` = active subcarriers.

---

## 1. Quick index

| Category | Functions |
| --- | --- |
| Random numbers | `np.random.default_rng`, `np.random.SeedSequence`, `rng.standard_normal`, `rng.integers`, `rng.random` |
| Array creation | `np.zeros`, `np.zeros_like`, `np.ones`, `np.full`, `np.eye`, `np.array`, `np.arange` |
| Joining / reshaping | `np.concatenate`, `.reshape`, `.T`, `.copy`, `.astype`, `np.asarray`, slicing, `[:, None]` |
| Math | `np.sqrt`, `np.abs`, `np.conj`, `np.log2`, `np.mean`, `np.std`, `np.sum`, `np.sort`, `np.argmin`, `np.nan` |
| Signal processing | `np.fft.fft`, `np.fft.ifft`, `np.convolve` |
| Linear algebra | `np.linalg.svd`, `np.linalg.lstsq`, `np.linalg.matrix_rank`, `np.outer`, `np.tensordot`, `@` |

---

## 2. MATLAB to NumPy conventions (read this first)

Most porting bugs come from these differences, not from the math.

| Topic | MATLAB | NumPy | Where it matters here |
| --- | --- | --- | --- |
| Index base | 1-based | 0-based | Subcarrier indices, `cyclic_prefix_length+1` becomes `cp` |
| Ranges | `a:b` is **inclusive** | `np.arange(a, b)` **excludes** `b` | `-26:-22` becomes `np.arange(-26, -21)` |
| Slices | `x(cp+1:cp+N)` | `x[cp:cp+N]` | CP removal |
| Last elements | `x(N-cp+1:N)` | `x[-cp:]` | CP insertion |
| `reshape` order | Column-major | Row-major (C) by default | Flattening decoded symbols (see Section 8) |
| Transpose | `.'` plain, `'` conjugate | `.T` plain only | `u*u'` becomes `np.outer(u, np.conj(u))` |
| Matrix multiply | `*` | `@` (`*` is elementwise) | Projection and equalizer |
| Elementwise ops | `.*`, `./`, `.^` | `*`, `/`, `**` | Everywhere |
| Not-equal | `~=` | `!=` | BER counting |
| Solve | `A\b` | `np.linalg.lstsq(A, b)[0]` | Zero-forcing equalizer |
| Dropping singleton dims | `squeeze` | Integer indexing drops axes | Per-tone snapshots |
| `std` default | Sample std (divides by N-1) | Population std (`ddof=0`) | Spatial interference std |
| Complex SVD / eig | LAPACK | LAPACK | Same algorithm family, but vector phase/sign is arbitrary in both |
| RNG | `randn`, `rand`, `randi` | `Generator` methods | Streams differ, so results match **statistically**, never sample by sample |

---

## 3. Random number generation

### `np.random.default_rng(seed)`
Creates a modern `Generator` (PCG64). All randomness (bits, channels, noise, jammer) comes from this one object, so the same `seed` reproduces the same run.
- **MATLAB:** `rng('shuffle')` (or `rng(seed)`).
- **Used in:** `run_simulation`.

### `np.random.SeedSequence().entropy`
Pulls fresh OS entropy and exposes it as an integer. It is logged and returned so a run can be reproduced.
- **MATLAB:** `rng('shuffle')`. The MATLAB comment says "Fixed seed for REPRODUCIBILITY" but `'shuffle'` is the opposite, and the seed is never recorded. The Python version is an improvement.
- **Used in:** `seed = np.random.SeedSequence().entropy`.

### `rng.standard_normal(shape)`
Gaussian samples, mean 0, variance 1.
- **MATLAB:** `randn(...)`.
- **Used in:** `_complex_normal`:
```python
np.sqrt(0.5) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
```
MATLAB: `sqrt(0.5)*randn(...) + 1j*sqrt(0.5)*randn(...)`. Identical: CN(0,1), used for noise, jammer, and Rayleigh channel taps.

### `rng.integers(low, high, size, dtype)`
Random integers in `[low, high)`.
- **MATLAB:** `randi([0 1], trials, symbols, tx, bps, tones)`. Note MATLAB's range is inclusive `[0 1]`, NumPy's is `[0, 2)`.
- `dtype=np.int8` stores each bit in one byte; MATLAB would use 8 bytes per `double`.

### `rng.random()`
One uniform float in `[0, 1)`.
- **MATLAB:** `rand()`.
- **Used in:** random jammer power, `10 ** (jammer_power_dB * rng.random() / 10)`.

> Not ported: MATLAB's `randperm(5)` (shuffling the five operator situations) would be `rng.permutation(5)`. The `TVWSOperatorButtonPushed` demo logic is outside the numerical core.

---

## 4. Array creation

| NumPy | MATLAB | Notes / where used |
| --- | --- | --- |
| `np.zeros(shape, dtype)` | `zeros(...)` | Buffers for `symbols_freq`, `legit`, BER accumulators, `received_by_tone`, `spatial_interference`, bit-mapping table |
| `np.zeros_like(a)` | `zeros(size(a))` | `comp`, `noncomp`, `jammer_noncompliant_frequency` |
| `np.ones(n)` | `ones(n,1)` | Fallback dominant vector in `_dominant_vector` |
| `np.full(shape, v)` | `NaN(shape)` | Live BER arrays pre-filled with `np.nan` so unfinished SNR points leave gaps in plots |
| `np.nan` | `NaN` | Plotting skips NaN points |
| `np.eye(n)` | `eye(n)` | Identity and projectors `P = I - v vᴴ` |
| `np.array(list, dtype)` | `[ ... ]` literal | Constellation point lists |
| `np.arange(a, b, s)` | `a:s:b` | **MATLAB inclusive, NumPy exclusive** |

`np.arange` in two roles:
- `ofdm_utils.create_used_subcarrier_indices`: `np.arange(-26, -21)` is MATLAB `-26:-22`.
- `_map_symbols`: `np.arange(bps-1, -1, -1)` gives the MSB-first shifts `[bps-1, ..., 0]`.

---

## 5. Joining, reshaping, copying, indexing

### `np.concatenate(arrays, axis)`
Joins arrays along an existing axis.
- **MATLAB:** `[a, b]` (horizontal) or `[a; b]` (vertical).
- **`_add_cp`:** `np.concatenate((x[..., -cp_len:], x), axis=-1)`. MATLAB: `[x(:, N-cp+1:N), x]`. Prepending the last `cp` samples *is* the cyclic prefix.
- **`create_used_subcarrier_indices`:** MATLAB `[-26:-22, -20:-8, -6:-1, 1:6, 8:20, 22:26]`.

### `array.reshape(-1)` / `np.asarray(x).reshape(-1)`
Flatten to 1-D. **MATLAB:** `reshape(x, [N,1])`, but MATLAB is column-major and NumPy row-major, so the element order can differ. The code compensates with `.T` first (Section 8).

### `array.T`
**MATLAB:** `.'`. For complex conjugate transpose use `.conj().T` (MATLAB `'`). Used for `y[:, used_idx].T` and `estimate.T`.

### `array.copy()`
NumPy slices are views, so characterization arrays are copied before the loop overwrites them. MATLAB assignment always copies, so there is no equivalent line.

### `array.astype(dtype)`
**MATLAB:** `double(x)`. Prevents int8 overflow in bit-weight math and makes indices integers.

### `np.asarray(x)`
Converts to ndarray without copying. No MATLAB equivalent needed.

### Slicing and broadcasting syntax
| NumPy | Meaning | MATLAB |
| --- | --- | --- |
| `x[..., -cp:]` | Last `cp` samples, any leading axes | `x(:, end-cp+1:end)` |
| `legit[:, start:stop]` | **CP removal** (`start=cp`, `stop=cp+n_fft`) | `x(:, cp+1:cp+N)` |
| `symbols_freq[:, used_idx] = s` | Assign active tones | `symbols_freq(:, used_idx) = s` |
| `symbols[:, None]`, `constellation[None, :]` | Add axis for broadcasting | `s*ones(1,M)`, `ones(N,1)*c` |
| `column_sum[column_sum == 0] = 1.0` | Boolean-mask assignment | `x(x==0) = 1` |
| `spatial /= column_sum[None, :]` | Column-wise divide | `A ./ sum(A,1)` |

---

## 6. Elementwise math and statistics

| NumPy | MATLAB | Role |
| --- | --- | --- |
| `np.sqrt(x)` | `sqrt` | Scaling: `√0.5`, `√n_fft`, `√jammer_power`, `√noise_variance`, constellation normalization |
| `np.abs(x)` | `abs` | Magnitude / energy `abs(x)**2` |
| `np.conj(x)` | `conj` (or `'`) | `v vᴴ` and the CP correlation |
| `np.log2(x)` | `log2` | Bits per symbol |
| `np.mean(x, axis)` | `mean(x, dim)` | Constellation energy, averaging metrics and spatial interference |
| `np.std(x, axis)` | `std(x)` | **Different default** (see Finding F3) |
| `np.sum(x, axis)` | `sum(x, dim)` | Error counts, energies, normalizers |
| `np.sort(a)` | `sort(a)` | Ascending subcarrier indices |
| `np.argmin(a, axis=1)` | `[~, i] = min(a, [], 2)` | Nearest constellation point (NumPy 0-based, MATLAB 1-based) |
| `10 ** (x/10)` | `10^(x/10)` | dB to linear |
| `(a + n//2 - 1) % n` | `mod(...)` | Wrap into `0..n_fft-1` |
| `1 << np.arange(...)` | `2.^(...)` or `bitshift` | Powers of two `[8,4,2,1]` |
| `a != b`, `np.sum(...)` | `sum(sum(a ~= b))` | Bit-error count |

Error-count idiom: `np.sum(true_bits != decoded)` matches MATLAB `sum(sum(true_bits~=bits_hat))`.

---

## 7. Signal-processing functions

### `np.fft.fft(x, n=None, axis=-1)`
Time to frequency. `n` zero-pads or truncates.
- **MATLAB:** `fft(x, N, dim)`. NumPy `axis=1` is MATLAB `dim=2`; NumPy `axis=-1` for a 3-D channel is MATLAB `dim=3`.
- **`_channel_to_frequency`:** `np.fft.fft(h, n=n_fft, axis=-1)/np.sqrt(n_fft)`. MATLAB: `(1/sqrt(N))*fft(h, N, 3)`.
- **`to_freq`:** `np.fft.fft(x, axis=1)/np.sqrt(n_fft)`. MATLAB: `(1/sqrt(N))*fft(x, N, 2)`.

### `np.fft.ifft(x, axis=-1)`
Frequency to time. **MATLAB:** `ifft`. Both libraries put the `1/N` on the inverse, so `np.sqrt(n_fft) * ifft(...)` is the same in both languages.

### `np.convolve(a, b)`
Full linear convolution, length `len(a)+len(b)-1`. **MATLAB:** `conv(a, b)`. Both are 1-D only, so the code loops over antennas. Output length `n_fft+cp+taps-1` matches MATLAB's `num_subcarriers+cyclic_prefix_length+num_channel_taps-1`.

---

## 8. Linear algebra

### `np.linalg.svd(A, full_matrices=False)`
`A = U Σ Vᴴ`. **MATLAB:** `[U,S,~] = svd(A)`; `S` is a matrix, while NumPy returns singular values as a vector, so `diag(S)` is not needed. `U[:, 0]` is the dominant spatial direction. Only the first column and `S[0]` are used, so `full_matrices=False` makes no difference to the result.

### `np.outer(a, b)`
**MATLAB:** `u*u'`. With `np.conj`, `np.outer(u, np.conj(u))` is the rank-1 projector, and `np.eye(n_rx) - np.outer(...)` nulls the jammer direction (MATLAB: `eye(n) - u*u'`).

### `@` (matrix multiplication)
**MATLAB:** `*`. Used for `pn @ snapshot`, `projector @ channel`, `projector @ received`.

### `np.linalg.matrix_rank(A)`
**MATLAB:** `rank(A)`. In `_lstsq_or_zeros`, rank 0 means an all-zero matrix, so zeros are returned. MATLAB has no equivalent guard.

### `np.linalg.lstsq(A, b, rcond=None)`
Solves `min ‖Ax − b‖²`. **MATLAB:** `A\b`. For well-conditioned square or tall systems the results agree. For rank-deficient systems NumPy returns the **minimum-norm** solution while MATLAB returns a *basic* solution (with a warning), so results can differ there.

### `np.tensordot(a, b, axes)`
**MATLAB:** `bi2de(bits_slice, 'left-msb') + 1` inside a per-tone `for` loop. NumPy contracts the bits axis with weights `[2^(b-1), ..., 1]` for **all tones and antennas at once** and returns **0-based** indices (no `+1`).

---

## 9. Section-by-section mapping (MATLAB to Python)

| MATLAB section | Python location | Equivalence |
| --- | --- | --- |
| Parameters (`num_receive_antennas`, `snr_dB_list = -5:1:30`) | `SimulationParameters`; `np.arange(-5.0, 31.0, 1.0)` | Same |
| `used_subcarrier_indices = [...]+32` | `create_used_subcarrier_indices` | Same tones (0-based); **48 tones** for 64 FFT |
| `switch (modulation_scheme)` | `create_constellation` | Same, **except the 64-QAM typo in MATLAB (F5)** |
| Constellation normalization | `/= np.sqrt(np.mean(np.abs(c)**2))` | Same |
| `de2bi(0:M-1, Q, 'left-msb')` | `create_bit_mapping` | Same (MSB first) |
| `randi([0 1], ...)` | `rng.integers(0, 2, ...)` | Same layout `(trial, symbol, tx, bit, tone)` |
| Jammer / user channel `sqrt(0.5)*randn + 1j*...` | `_complex_normal` | Same distribution |
| `fft(h, N, dim)/sqrt(N)` | `_channel_to_frequency` | Same. `jammer_channel_freq` is unused in both |
| `rho = tx*10^(dB/10)` or `...*rand()` | Same expression with `rng.random()` | Same |
| `noise_variance = taps*tx*10^(-snr/10)` | Same expression | Same |
| Jammer OFDM-compliant: `sqrt(N)*ifft`, add CP, scale | `_generate_symbol_signals` | Same |
| Jammer non-compliant: time-domain noise of length `N+CP` | `_complex_normal(rng, n_fft + cp)` | Same |
| `bi2de` loop, `symbols_freq(:,used)=...`, IFFT, CP | `_map_symbols`, `_add_cp` | Same (vectorized) |
| `conv(...)` summed over TX | `np.convolve` in nested loop | Same |
| Truncate CP (`cp+1:cp+N`), add noise | `[:, start:stop] + noise` | Same (jammer-only gets no noise in both) |
| Frequency-domain `fft/√N`, tone selection | `to_freq`, `[:, used_idx].T` | Same |
| SVD per tone, `P = I - u u'` | `_dominant_vector`, `np.outer` | Same idea; **SISO handling differs (F6)** |
| `y./H` (legit), `(√N·P·H)\(P·y)` (others) | `_lstsq_or_zeros`, `channel_scale` | Same structure; **scaling issue F1** |
| `min(abs(...).^2,[],2)`, `bit_mappings(idx,:)` | `minimum_distance_decode` | Same (0-based index) |
| `sum(sum(true_bits ~= bits_hat))` | `np.sum(true_bits != decoded)` | Same |
| BER normalization | `/ total_bits` | Same |
| `res.spatial_interference... ./ sum(...,1)`, `std`, `mean` | Same | **Different std (F3), different accumulation (F4)** |
| `writetable(...)` | `DataFrame.to_csv` | Column names differ (F7) |
| `avg_y_*` accumulation | Not ported | See F9 |
| Plots (`plot`, `semilogy`, `imagesc`) | `live_callback` payloads | Plot code lives in the GUI, not in these files |

---

## 10. Differences and issues found

Severity: **High** = changes numerical results, **Medium** = affects outputs or MIMO use, **Low** = cosmetic or documentation.

| ID | Severity | Topic | MATLAB | Python | Suggested action |
| --- | --- | --- | --- | --- | --- |
| F1 | High (16/64-QAM) | Legitimate-signal scaling | `y ./ H`, with no `√N` | `channel_scale = 1.0` for the legitimate case | See details below |
| F2 | Medium | Live CP correlation input | n/a (metrics are new) | Live update passes `legit[:, start:stop]`, which has **no CP** | Pass the full signal as the characterization call does |
| F3 | Low-Medium | Std dev | `std` divides by N-1 | `np.std` divides by N | Use `np.std(..., ddof=1)` to match |
| F4 | Medium (MIMO only) | Spatial interference accumulation | Adds `dominant_sv` to **every** antenna | Adds `abs(vec_n)**2 * singular_values[0]` | Decide which is intended; identical for `n_rx = 1` |
| F5 | Medium (64-QAM only) | 64-QAM constellation | Last row contains `-3-3i` instead of `+3-3i` (duplicate point, missing point) | Correct, symmetric set | Fix the MATLAB typo; Python is right |
| F6 | Verify | `n_rx = 1` projection | `eye(1) - u*u'` from SVD of `squeeze(...)` | Forces identity projector | Run `size(projection_matrix_compliant)` in MATLAB |
| F7 | Low | Projection BER | No-op projector; hard-coded 2 CSV columns `BER_Null1Dim`, `BER_Null2Dim` | Loop ignores `projection_index`; columns `BER_Null1..BER_Null{taps}` | Rename columns if downstream tools expect MATLAB names |
| F8 | Low | Subcarrier index base | 1-based (6..58) | 0-based (5..57) | Add `+1` when plotting against MATLAB axes |
| F9 | Medium | Averaged signals | `avg_y_*` computed for the "Avg over Trial" tab | Not computed or returned | Add accumulation if those plots are needed |
| F10 | Low | Characterization payloads | Plots jammer-only tones for the two jammer panels | Final update matches; the in-loop live update sends **total** signals | Make both consistent |
| F11 | Low | `_lstsq_or_zeros` | `\` has no guard | `matrix_rank > 0` only catches an all-zero matrix | Fine, but docstring overstates it |

### F1: legitimate-path scaling (details)
With `s_time = √N·ifft(S)` and `y = fft(h⊛s)/√N`, the received tone is `y[k] = √N · H_f[k] · S[k]`, where `H_f` is the `1/√N`-normalized channel. The compliant and non-compliant paths use `√N·H` and recover `S` correctly. The legitimate path divides by `H` alone and recovers `√N · S` (8× too large for N = 64).
- **BPSK/QPSK:** the nearest-neighbor decision depends only on the sign, so the BER is unaffected.
- **16/64-QAM:** symbols are scaled outside the constellation, so decisions saturate to the outer ring. The legitimate (jammerless) BER curve is wrong.
- **Fix in Python:** use `np.sqrt(n_fft)` as `channel_scale` for `"legitimate"`. In MATLAB: `y ./ (sqrt(N)*H)`.

### F6: `n_rx = 1` projection
For `n_rx = 1`, indexing `jammer_compliant_freq_domain(trial, tone, :, :)` gives a 1×1×1×50 array, and `squeeze` turns that into a **50×1** column. The SVD would then run along the OFDM-symbol axis (U is 50×50) and `eye(1) - u*u'` would be a 50×50 matrix, not a scalar. The Python port treats the snapshot as `(n_rx, n_sym) = (1, 50)` and its comment assumes a scalar projector. If MATLAB really produces a 50×50 matrix, the two implementations compute different things for SISO. This should be confirmed by running one iteration in MATLAB.

---

## 11. Verified equivalent (no action needed)

- Parameter defaults and SNR sweep (`-5:1:30` and `np.arange(-5, 31, 1)`).
- Noise variance, jammer power, CP insertion/removal indices, FFT/IFFT scaling.
- Bit layout `(trial, symbol, tx, bit, tone)` and MSB-first mapping.
- Decoded-symbol and `true_bits` ordering (symbol, then TX, then bit) for the same number of transmit antennas.
- Tie-breaking in nearest-neighbor decoding (both pick the first minimum).
- BER denominators and per-trial, per-SNR accumulation order (SVD done inside the SNR loop, so interference accumulates over trials × SNRs in both).

---

## 12. Python additions with no MATLAB counterpart

| Feature | Location |
| --- | --- |
| `cp_correlation`, `spectral_concentration` detector metrics | `metrics.py` |
| Jammer energy before/after projection, "% removed" | `run_simulation` |
| Seed recording, reproducible runs | `run_simulation` |
| Stop, progress, and throttled live callbacks | `_should_emit`, callbacks |
| Rank-deficiency guard | `_lstsq_or_zeros` |
| Non-64 FFT sizes | `create_used_subcarrier_indices` (MATLAB hard-codes 64) |

### `cp_correlation(signal, cp_len, fft_size)`
```python
cp     = signal[:, :cp_len]                    # prefix
useful = signal[:, cp_len:cp_len + fft_size]   # FFT window
tail   = useful[:, -cp_len:]                   # last cp_len samples of the symbol
```
| NumPy call | Role |
| --- | --- |
| Slicing | Isolates CP and symbol tail per antenna |
| `np.conj(tail)` | Complex correlation |
| `np.sum(cp * np.conj(tail), axis=1)` | Numerator |
| `np.abs(...)**2`, `np.sum(..., axis=1)` | Energies |
| `np.sqrt(E_cp * E_tail) + 1e-12` | Normalizer (epsilon avoids divide-by-zero) |
| `np.abs(num/den)` | Normalized value in `[0, 1]` (Cauchy-Schwarz) |

A legitimate OFDM symbol gives a value near 1 because the CP is a copy of the tail. Noise-like jammers give values near 0.

### `spectral_concentration(freq, used_subcarrier_indices)`
| NumPy call | Role |
| --- | --- |
| `freq[:, used_idx]` | Fancy indexing of active bins |
| `np.sum(np.abs(freq)**2, axis=1)` | Total energy per antenna |
| `np.sum(np.abs(used)**2, axis=1)` | Active-band energy |
| `used / (total + 1e-12)` | Fraction in the allocated band |

---

## 13. Project function to NumPy call table

| Function | NumPy calls |
| --- | --- |
| `_complex_normal` | `np.sqrt`, `rng.standard_normal` |
| `_channel_to_frequency` | `np.fft.fft`, `np.sqrt` |
| `_add_cp` | `np.concatenate` |
| `_map_symbols` | `np.arange`, `np.tensordot`, `.astype`, bit shift |
| `_generate_symbol_signals` | `np.fft.ifft`, `np.zeros`, `np.zeros_like`, `np.convolve`, `np.sqrt` |
| `_lstsq_or_zeros` | `np.linalg.matrix_rank`, `np.linalg.lstsq`, `np.zeros`, `np.asarray` |
| `_dominant_vector` | `np.linalg.svd`, `np.ones`, `np.sqrt` |
| `run_simulation` | `default_rng`, `SeedSequence`, `rng.integers`, `rng.random`, `np.zeros`, `np.full`, `np.nan`, `np.eye`, `np.outer`, `np.conj`, `np.abs`, `np.sum`, `np.mean`, `np.std`, `np.fft.fft`, `.copy`, `.T`, `.reshape`, `@` |
| `create_used_subcarrier_indices` | `np.concatenate`, `np.arange`, `%`, `.astype`, `np.sort` |
| `create_constellation` | `np.array`, `np.sqrt`, `np.mean`, `np.abs` |
| `create_bit_mapping` | `np.log2`, `np.zeros`, shifts, `& 1` |
| `minimum_distance_decode` | `np.asarray`, `.reshape`, broadcasting, `np.abs`, `np.argmin` |
| `cp_correlation`, `spectral_concentration` | slicing, `np.conj`, `np.abs`, `np.sqrt`, `np.sum` |

Active subcarrier count for `n_fft = 64`: 5 + 13 + 6 + 6 + 13 + 5 = **48** tones.

---

## 14. Shape walk-through of one OFDM symbol

```text
bits[trial, symbol]              (n_tx, bits_per_symbol, n_used)
  └─ _map_symbols (tensordot) →  (n_tx, n_used)            indices → complex symbols
  └─ symbols_freq[:, used_idx]   (n_tx, n_fft)             active bins, zeros elsewhere
  └─ ifft + _add_cp →            (n_tx, n_fft + cp)
  └─ convolve with channel →     (n_rx, n_fft + cp + taps − 1)
  └─ [:, start:stop] + noise →   (n_rx, n_fft)             CP removed
  └─ fft / √n_fft →              (n_rx, n_fft)
  └─ [:, used_idx].T →           (n_used, n_rx)            per-tone snapshot
  └─ lstsq equalizer →           (n_tx, n_symbols)
  └─ .T.reshape(-1), argmin →    bits, compared with true_bits for BER
```

Why `.T` before `.reshape(-1)`: MATLAB's column-major `reshape` of an `n_tx × n_symbols` matrix yields symbol-major order (all TX for symbol 1, then symbol 2, ...). NumPy's row-major `reshape` would give TX-major order, so the transpose restores the MATLAB ordering. The ordering matches `bits[trial, :, :, :, tone].reshape(-1)` (symbol, TX, bit).
