# AOTS2Action Scalability Experiment

Marker: **REAL_HUMANITARIAN_GEOSPATIAL_DATA**.

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
| 5 | 0.10 | 286 | 0.000257 | 0.000255 | 0.000005 | 0.0360 | 5556271.11 |
| 5 | 0.25 | 716 | 0.000364 | 0.000364 | 0.000001 | 0.0361 | 9827046.05 |
| 5 | 0.50 | 1432 | 0.000507 | 0.000507 | 0.000002 | 0.0363 | 14119481.86 |
| 5 | 0.75 | 2147 | 0.000649 | 0.000644 | 0.000009 | 0.0366 | 16548191.94 |
| 5 | 1.00 | 2863 | 0.000787 | 0.000785 | 0.000005 | 0.0369 | 18191606.98 |
| 10 | 0.10 | 286 | 0.000400 | 0.000400 | 0.000001 | 0.0369 | 7154288.90 |
| 10 | 0.25 | 716 | 0.000579 | 0.000574 | 0.000011 | 0.0369 | 12367532.74 |
| 10 | 0.50 | 1432 | 0.000819 | 0.000808 | 0.000020 | 0.0370 | 17502232.17 |
| 10 | 0.75 | 2147 | 0.001045 | 0.001037 | 0.000021 | 0.0370 | 20543862.45 |
| 10 | 1.00 | 2863 | 0.001258 | 0.001255 | 0.000007 | 0.0370 | 22765576.88 |
| 20 | 0.10 | 286 | 0.000691 | 0.000688 | 0.000009 | 0.0370 | 8281431.83 |
| 20 | 0.25 | 716 | 0.000995 | 0.000995 | 0.000003 | 0.0370 | 14391479.75 |
| 20 | 0.50 | 1432 | 0.001419 | 0.001411 | 0.000016 | 0.0370 | 20187247.22 |
| 20 | 0.75 | 2147 | 0.001815 | 0.001803 | 0.000023 | 0.0370 | 23662967.35 |
| 20 | 1.00 | 2863 | 0.002224 | 0.002204 | 0.000044 | 0.0370 | 25757774.61 |
| 40 | 0.10 | 286 | 0.001264 | 0.001260 | 0.000009 | 0.0372 | 9049324.90 |
| 40 | 0.25 | 716 | 0.001832 | 0.001825 | 0.000022 | 0.0372 | 15637435.36 |
| 40 | 0.50 | 1432 | 0.002619 | 0.002610 | 0.000022 | 0.0369 | 21871937.39 |
| 40 | 0.75 | 2147 | 0.003355 | 0.003345 | 0.000023 | 0.0327 | 25599059.71 |
| 40 | 1.00 | 2863 | 0.004105 | 0.004090 | 0.000030 | 0.0315 | 27896270.25 |
| M_full | 0.10 | 286 | 0.001577 | 0.001573 | 0.000012 | 0.0314 | 9247954.14 |
| M_full | 0.25 | 716 | 0.002291 | 0.002284 | 0.000017 | 0.0314 | 15941510.02 |
| M_full | 0.50 | 1432 | 0.003276 | 0.003269 | 0.000019 | 0.0314 | 22295517.37 |
| M_full | 0.75 | 2147 | 0.004203 | 0.004188 | 0.000034 | 0.0314 | 26051456.42 |
| M_full | 1.00 | 2863 | 0.005151 | 0.005122 | 0.000049 | 0.0314 | 28348669.65 |

## Scaling Analysis

Runtime versus ensemble size at fixed spatial volume:
- x_size=286: log-log slope 0.788, R^2 0.996
- x_size=716: log-log slope 0.797, R^2 0.997
- x_size=1432: log-log slope 0.808, R^2 0.997
- x_size=2147: log-log slope 0.809, R^2 0.997
- x_size=2863: log-log slope 0.816, R^2 0.997

Runtime versus spatial volume at fixed ensemble size:
- m_size=5: log-log slope 0.481, R^2 0.987
- m_size=10: log-log slope 0.495, R^2 0.991
- m_size=20: log-log slope 0.503, R^2 0.989
- m_size=40: log-log slope 0.507, R^2 0.990
- m_size=51: log-log slope 0.509, R^2 0.989

Peak memory versus ensemble size:
- Peak RSS is dominated by the loaded Python process and real dataset. The measured peak is nearly flat across M because the implementation streams ensemble members over the grid instead of materializing an M x |X| distance matrix.

Throughput across all configurations:
- The global log-log runtime slope versus M*|X| is 0.657 (R^2 0.944).
- Observed scaling is not cleanly consistent with O(M|X|) under the stated real-kernel protocol.

## Largest Configuration

- Configuration: M_full = 51, N_X = 2863
- Runtime T: 0.005151 s per forecast case
- Peak memory: 0.0314 GB
- Throughput: 28348669.65 items/s
- Mean runtime: 0.005151 s
- Median runtime: 0.005122 s
- Completes within one 6 h forecast cycle: yes

## Figure-Ready Data

- Runtime vs M: `results_AOTS2Action/csv/rq4_runtime_vs_m_REAL.csv`
- Runtime vs |X|: `results_AOTS2Action/csv/rq4_runtime_vs_x_REAL.csv`
- Throughput: `results_AOTS2Action/csv/rq4_throughput_REAL.csv`
- Peak memory: `results_AOTS2Action/csv/rq4_peak_memory_REAL.csv`
- Raw repeated runs: `results_AOTS2Action/csv/rq4_scalability_raw_REAL.csv`
- Slopes: `results_AOTS2Action/csv/rq4_scalability_slopes_REAL.csv`

Do not state that scaling is linear beyond this measured real setting.
