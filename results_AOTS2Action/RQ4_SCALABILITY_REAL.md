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
| 5 | 0.10 | 286 | 0.000260 | 0.000259 | 0.000002 | 0.0364 | 5508050.34 |
| 5 | 0.25 | 716 | 0.000371 | 0.000366 | 0.000015 | 0.0365 | 9675421.61 |
| 5 | 0.50 | 1432 | 0.000515 | 0.000510 | 0.000018 | 0.0367 | 13908329.17 |
| 5 | 0.75 | 2147 | 0.000645 | 0.000645 | 0.000002 | 0.0369 | 16637041.61 |
| 5 | 1.00 | 2863 | 0.000790 | 0.000787 | 0.000007 | 0.0373 | 18120407.67 |
| 10 | 0.10 | 286 | 0.000402 | 0.000402 | 0.000001 | 0.0373 | 7113217.54 |
| 10 | 0.25 | 716 | 0.000579 | 0.000577 | 0.000005 | 0.0373 | 12369817.58 |
| 10 | 0.50 | 1432 | 0.000812 | 0.000812 | 0.000003 | 0.0373 | 17633062.36 |
| 10 | 0.75 | 2147 | 0.001040 | 0.001039 | 0.000008 | 0.0373 | 20642803.08 |
| 10 | 1.00 | 2863 | 0.001257 | 0.001256 | 0.000006 | 0.0373 | 22768101.94 |
| 20 | 0.10 | 286 | 0.000708 | 0.000694 | 0.000026 | 0.0374 | 8084851.06 |
| 20 | 0.25 | 716 | 0.001018 | 0.001013 | 0.000022 | 0.0374 | 14068145.62 |
| 20 | 0.50 | 1432 | 0.001425 | 0.001418 | 0.000016 | 0.0374 | 20098265.93 |
| 20 | 0.75 | 2147 | 0.001817 | 0.001812 | 0.000010 | 0.0374 | 23637591.11 |
| 20 | 1.00 | 2863 | 0.002218 | 0.002210 | 0.000016 | 0.0374 | 25820358.95 |
| 40 | 0.10 | 286 | 0.001282 | 0.001269 | 0.000023 | 0.0376 | 8929255.69 |
| 40 | 0.25 | 716 | 0.001837 | 0.001835 | 0.000007 | 0.0376 | 15589389.89 |
| 40 | 0.50 | 1432 | 0.002622 | 0.002620 | 0.000008 | 0.0376 | 21848250.60 |
| 40 | 0.75 | 2147 | 0.003374 | 0.003362 | 0.000031 | 0.0311 | 25453012.45 |
| 40 | 1.00 | 2863 | 0.004116 | 0.004096 | 0.000037 | 0.0308 | 27824381.21 |
| M_full | 0.10 | 286 | 0.001589 | 0.001586 | 0.000010 | 0.0311 | 9179696.68 |
| M_full | 0.25 | 716 | 0.002300 | 0.002299 | 0.000005 | 0.0311 | 15875397.68 |
| M_full | 0.50 | 1432 | 0.003288 | 0.003278 | 0.000028 | 0.0311 | 22212175.71 |
| M_full | 0.75 | 2147 | 0.004207 | 0.004205 | 0.000008 | 0.0311 | 26026484.81 |
| M_full | 1.00 | 2863 | 0.005142 | 0.005135 | 0.000022 | 0.0311 | 28394696.40 |

## Scaling Analysis

Runtime versus ensemble size at fixed spatial volume:
- x_size=286: log-log slope 0.790, R^2 0.996
- x_size=716: log-log slope 0.793, R^2 0.997
- x_size=1432: log-log slope 0.805, R^2 0.996
- x_size=2147: log-log slope 0.813, R^2 0.997
- x_size=2863: log-log slope 0.814, R^2 0.997

Runtime versus spatial volume at fixed ensemble size:
- m_size=5: log-log slope 0.476, R^2 0.989
- m_size=10: log-log slope 0.491, R^2 0.990
- m_size=20: log-log slope 0.490, R^2 0.989
- m_size=40: log-log slope 0.503, R^2 0.988
- m_size=51: log-log slope 0.506, R^2 0.990

Peak memory versus ensemble size:
- Peak RSS is dominated by the loaded Python process and real dataset. The measured peak is nearly flat across M because the implementation streams ensemble members over the grid instead of materializing an M x |X| distance matrix.

Throughput across all configurations:
- The global log-log runtime slope versus M*|X| is 0.654 (R^2 0.942).
- Observed scaling is not cleanly consistent with O(M|X|) under the stated real-kernel protocol.

## Largest Configuration

- Configuration: M_full = 51, N_X = 2863
- Runtime T: 0.005142 s per forecast case
- Peak memory: 0.0311 GB
- Throughput: 28394696.40 items/s
- Mean runtime: 0.005142 s
- Median runtime: 0.005135 s
- Completes within one 6 h forecast cycle: yes

## Figure-Ready Data

- Runtime vs M: `results_AOTS2Action/csv/rq4_runtime_vs_m_REAL.csv`
- Runtime vs |X|: `results_AOTS2Action/csv/rq4_runtime_vs_x_REAL.csv`
- Throughput: `results_AOTS2Action/csv/rq4_throughput_REAL.csv`
- Peak memory: `results_AOTS2Action/csv/rq4_peak_memory_REAL.csv`
- Raw repeated runs: `results_AOTS2Action/csv/rq4_scalability_raw_REAL.csv`
- Slopes: `results_AOTS2Action/csv/rq4_scalability_slopes_REAL.csv`

Do not state that scaling is linear beyond this measured real setting.
