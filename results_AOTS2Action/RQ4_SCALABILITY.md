# AOTS2Action Scalability Experiment

Marker: **PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE**.

Runtime is reported per forecast case. Each repeat measures the full eligible-case batch, then divides by the fixed eligible case count so throughput is exactly `M * |X| / runtime` for one operational forecast case.

## Environment

- CPU model: Apple M1 Pro
- RAM: 16.00 GB
- Operating system: macOS-26.5.2-arm64-arm-64bit
- Python version: 3.9.6
- NumPy version: 2.0.2

## Complete M x |X| Results Table

| M | X fraction | X size | mean runtime s | median runtime s | runtime std s | peak memory GB | throughput items/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.10 | 286 | 0.000268 | 0.000267 | 0.000003 | 0.0343 | 5341473.47 |
| 5 | 0.25 | 716 | 0.000374 | 0.000374 | 0.000002 | 0.0315 | 9566660.12 |
| 5 | 0.50 | 1432 | 0.000523 | 0.000520 | 0.000006 | 0.0317 | 13699923.00 |
| 5 | 0.75 | 2147 | 0.000669 | 0.000661 | 0.000023 | 0.0319 | 16059059.80 |
| 5 | 1.00 | 2863 | 0.000824 | 0.000819 | 0.000024 | 0.0310 | 17382210.90 |
| 10 | 0.10 | 286 | 0.000415 | 0.000415 | 0.000003 | 0.0311 | 6885273.29 |
| 10 | 0.25 | 716 | 0.000591 | 0.000591 | 0.000002 | 0.0311 | 12106293.36 |
| 10 | 0.50 | 1432 | 0.000837 | 0.000829 | 0.000023 | 0.0311 | 17123985.95 |
| 10 | 0.75 | 2147 | 0.001059 | 0.001059 | 0.000002 | 0.0311 | 20277333.65 |
| 10 | 1.00 | 2863 | 0.001287 | 0.001283 | 0.000009 | 0.0311 | 22250925.12 |
| 20 | 0.10 | 286 | 0.000713 | 0.000707 | 0.000017 | 0.0309 | 8026933.22 |
| 20 | 0.25 | 716 | 0.001028 | 0.001026 | 0.000005 | 0.0308 | 13931631.93 |
| 20 | 0.50 | 1432 | 0.001462 | 0.001453 | 0.000027 | 0.0308 | 19593514.57 |
| 20 | 0.75 | 2147 | 0.001852 | 0.001848 | 0.000010 | 0.0307 | 23182687.57 |
| 20 | 1.00 | 2863 | 0.002254 | 0.002252 | 0.000009 | 0.0307 | 25401932.68 |
| 40 | 0.10 | 286 | 0.001301 | 0.001297 | 0.000015 | 0.0309 | 8797257.06 |
| 40 | 0.25 | 716 | 0.001876 | 0.001876 | 0.000009 | 0.0309 | 15267859.59 |
| 40 | 0.50 | 1432 | 0.002677 | 0.002673 | 0.000010 | 0.0310 | 21397181.49 |
| 40 | 0.75 | 2147 | 0.003454 | 0.003436 | 0.000074 | 0.0310 | 24873941.33 |
| 40 | 1.00 | 2863 | 0.004186 | 0.004185 | 0.000029 | 0.0309 | 27356142.86 |
| M_full | 0.10 | 286 | 0.001619 | 0.001617 | 0.000007 | 0.0300 | 9010921.02 |
| M_full | 0.25 | 716 | 0.002350 | 0.002345 | 0.000015 | 0.0301 | 15536753.13 |
| M_full | 0.50 | 1432 | 0.003359 | 0.003349 | 0.000026 | 0.0301 | 21742982.80 |
| M_full | 0.75 | 2147 | 0.004295 | 0.004293 | 0.000016 | 0.0301 | 25496652.68 |
| M_full | 1.00 | 2863 | 0.005243 | 0.005239 | 0.000020 | 0.0301 | 27848488.75 |

## Scaling Analysis

Runtime versus ensemble size at fixed spatial volume:
- x_size=286: log-log slope 0.782, R^2 0.996
- x_size=716: log-log slope 0.797, R^2 0.997
- x_size=1432: log-log slope 0.806, R^2 0.997
- x_size=2147: log-log slope 0.809, R^2 0.996
- x_size=2863: log-log slope 0.805, R^2 0.996

Runtime versus spatial volume at fixed ensemble size:
- m_size=5: log-log slope 0.482, R^2 0.984
- m_size=10: log-log slope 0.487, R^2 0.989
- m_size=20: log-log slope 0.496, R^2 0.991
- m_size=40: log-log slope 0.505, R^2 0.989
- m_size=51: log-log slope 0.506, R^2 0.990

Peak memory versus ensemble size:
- Peak RSS is dominated by the loaded Python process and proxy dataset. The measured peak is nearly flat across M because the implementation streams ensemble members over the grid instead of materializing an M x |X| distance matrix.

Throughput across all configurations:
- The global log-log runtime slope versus M*|X| is 0.654 (R^2 0.943).
- Observed scaling is not cleanly consistent with O(M|X|) under the stated proxy-kernel protocol.

## Largest Configuration

- Configuration: M_full = 51, N_X = 2863
- Runtime T: 0.005243 s per forecast case
- Peak memory: 0.0301 GB
- Throughput: 27848488.75 items/s
- Mean runtime: 0.005243 s
- Median runtime: 0.005239 s
- Completes within one 6 h forecast cycle: yes

## Figure-Ready Data

- Runtime vs M: `results_AOTS2Action/csv/rq4_runtime_vs_m_PROXY.csv`
- Runtime vs |X|: `results_AOTS2Action/csv/rq4_runtime_vs_x_PROXY.csv`
- Throughput: `results_AOTS2Action/csv/rq4_throughput_PROXY.csv`
- Peak memory: `results_AOTS2Action/csv/rq4_peak_memory_PROXY.csv`
- Raw repeated runs: `results_AOTS2Action/csv/rq4_scalability_raw_PROXY.csv`
- Slopes: `results_AOTS2Action/csv/rq4_scalability_slopes_PROXY.csv`

Do not state that scaling is linear beyond this measured proxy setting.
