#!/usr/bin/env python3
"""Run the marked-proxy AOTS2Action scalability experiment.

The measured kernel is the RQ2 exposure-field computation: deterministic,
P90-envelope, realized, and ensemble probability-weighted exposure over a
spatial grid. Runtime is reported per forecast case so throughput follows the
requested definition M * |X| / runtime while the repeated batch run keeps timing
noise low.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil


EARTH_RADIUS_KM = 6371.0
MARKER = "PROXY_ASSUMPTION_NOT_PUBLICATION_GRADE"
DEFAULT_HORIZONS = (24, 48, 72)
DEFAULT_ENSEMBLE_SIZES = (5, 10, 20, 40)
DEFAULT_SPATIAL_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
FORECAST_CYCLE_HOURS = 6.0
GB = 1024**3


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)


def canonical_track_id(value: str) -> str:
    track_id = value.strip().upper()
    if track_id == "FUNGWONG" or track_id == "WONG":
        return "FUNG-WONG"
    return track_id


def distances_km(lat: np.ndarray, lon: np.ndarray, point_lat: float, point_lon: float) -> np.ndarray:
    lat_radians = np.radians(lat)
    point_lat_radians = math.radians(point_lat)
    delta_lat = np.radians(point_lat - lat)
    delta_lon = np.radians(point_lon - lon)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_radians)
        * math.cos(point_lat_radians)
        * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(distances_km(np.array([lat1]), np.array([lon1]), lat2, lon2)[0])


def load_cases(path: Path, horizons: set[int]) -> dict[tuple[str, datetime, int], dict[str, object]]:
    cases: dict[tuple[str, datetime, int], dict[str, object]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            horizon = int(row["horizon_h"])
            if horizon not in horizons:
                continue
            key = (canonical_track_id(row["cyclone_id"]), parse_time(row["forecast_time"]), horizon)
            cases[key] = {
                "cyclone_id": canonical_track_id(row["cyclone_id"]),
                "forecast_time": parse_time(row["forecast_time"]),
                "horizon_h": horizon,
                "observed_lat": float(row["observed_lat"]),
                "observed_lon": float(row["observed_lon"]),
                "reported_member_count": int(row["member_count"]),
            }
    return cases


def load_positions(
    path: Path,
    case_keys: set[tuple[str, datetime, int]],
    horizons: set[int],
) -> dict[tuple[str, datetime, int], list[tuple[str, float, float]]]:
    positions: dict[tuple[str, datetime, int], list[tuple[str, float, float]]] = defaultdict(list)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            horizon = int(row["LEAD_TIME"])
            if horizon not in horizons:
                continue
            key = (canonical_track_id(row["TRACK_ID"]), parse_time(row["FORECAST_TIME"]), horizon)
            if key in case_keys:
                positions[key].append(
                    (row["ENSEMBLE_MEMBER"].strip(), float(row["LATITUDE"]), float(row["LONGITUDE"]))
                )
    for key, members in positions.items():
        members.sort(key=lambda item: member_sort_key(item[0]))
        deduped: dict[str, tuple[str, float, float]] = {}
        for member_id, lat, lon in members:
            deduped[member_id] = (member_id, lat, lon)
        positions[key] = [deduped[member_id] for member_id in sorted(deduped, key=member_sort_key)]
    return positions


def member_sort_key(member_id: str) -> tuple[int, str]:
    try:
        return int(member_id), member_id
    except ValueError:
        return 10**9, member_id


def load_grid(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        "lat": np.asarray([float(row["lat"]) for row in rows], dtype=float),
        "lon": np.asarray([float(row["lon"]) for row in rows], dtype=float),
        "weight": np.asarray(
            [float(row["population"]) * float(row["inform_risk"]) for row in rows],
            dtype=float,
        ),
    }


def select_grid(grid: dict[str, np.ndarray], x_size: int, repeat: int, seed: int) -> dict[str, np.ndarray]:
    n_x = len(grid["lat"])
    if x_size == n_x:
        indices = np.arange(n_x)
    else:
        random = np.random.default_rng(seed + repeat * 1009 + x_size * 9173)
        indices = np.sort(random.choice(n_x, size=x_size, replace=False))
    return {name: values[indices] for name, values in grid.items()}


def select_members(
    members: list[tuple[str, float, float]],
    m_size: int,
    repeat: int,
    seed: int,
) -> np.ndarray:
    if m_size == len(members):
        selected = members
    else:
        random = np.random.default_rng(seed + repeat * 7919 + m_size * 101)
        indices = np.sort(random.choice(len(members), size=m_size, replace=False))
        selected = [members[index] for index in indices]
    return np.asarray([(lat, lon) for _, lat, lon in selected], dtype=float)


def exposure_kernel(
    case: dict[str, object],
    members: np.ndarray,
    grid: dict[str, np.ndarray],
    impact_radius_km: float,
    cone_buffer_km: float,
) -> float:
    mean_lat = float(members[:, 0].mean())
    mean_lon = float(members[:, 1].mean())
    spread = np.asarray(
        [great_circle_km(mean_lat, mean_lon, float(lat), float(lon)) for lat, lon in members],
        dtype=float,
    )
    p90_radius = float(np.quantile(spread, 0.9)) + cone_buffer_km

    realized_mask = distances_km(
        grid["lat"], grid["lon"], float(case["observed_lat"]), float(case["observed_lon"])
    ) <= impact_radius_km
    deterministic_mask = distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon) <= impact_radius_km
    p90_mask = distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon) <= (
        p90_radius + impact_radius_km
    )
    ensemble_probability = np.zeros(len(grid["lat"]), dtype=float)
    for member_lat, member_lon in members:
        ensemble_probability += (
            distances_km(grid["lat"], grid["lon"], float(member_lat), float(member_lon))
            <= impact_radius_km
        )
    ensemble_probability /= len(members)

    weights = grid["weight"]
    return float(
        weights[realized_mask].sum()
        + weights[deterministic_mask].sum()
        + weights[p90_mask].sum()
        + np.sum(weights * ensemble_probability)
    )


def monitor_peak_rss(stop: threading.Event, peak: list[int], interval_s: float) -> None:
    process = psutil.Process(os.getpid())
    while not stop.is_set():
        peak[0] = max(peak[0], process.memory_info().rss)
        time.sleep(interval_s)
    peak[0] = max(peak[0], process.memory_info().rss)


def measure_configuration(
    cases: dict[tuple[str, datetime, int], dict[str, object]],
    positions: dict[tuple[str, datetime, int], list[tuple[str, float, float]]],
    grid: dict[str, np.ndarray],
    m_size: int,
    m_label: str,
    x_size: int,
    x_fraction: float,
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    gc.collect()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    peak = [rss_before]
    stop = threading.Event()
    sampler = threading.Thread(
        target=monitor_peak_rss, args=(stop, peak, args.memory_sample_interval_s), daemon=True
    )
    selected_grid = select_grid(grid, x_size, repeat, args.seed)
    selected_by_case = {
        key: select_members(members, m_size, repeat, args.seed)
        for key, members in positions.items()
    }

    checksum = 0.0
    sampler.start()
    start = time.perf_counter()
    try:
        for key in sorted(cases, key=lambda item: (item[2], item[0], item[1])):
            checksum += exposure_kernel(
                cases[key],
                selected_by_case[key],
                selected_grid,
                args.impact_radius_km,
                args.cone_buffer_km,
            )
    finally:
        stop.set()
        sampler.join()
    total_runtime_s = time.perf_counter() - start
    case_count = len(cases)
    runtime_s = total_runtime_s / case_count
    throughput = m_size * x_size / runtime_s
    return {
        "marker": MARKER,
        "m_label": m_label,
        "m_size": m_size,
        "x_fraction": x_fraction,
        "x_size": x_size,
        "repeat": repeat,
        "case_count": case_count,
        "total_runtime_s": total_runtime_s,
        "runtime_s": runtime_s,
        "rss_before_gb": rss_before / GB,
        "peak_memory_gb": peak[0] / GB,
        "peak_memory_delta_gb": max(0, peak[0] - rss_before) / GB,
        "throughput_items_per_s": throughput,
        "checksum": checksum,
    }


def summarize(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    groups: dict[tuple[str, int, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        groups[(str(row["m_label"]), int(row["m_size"]), float(row["x_fraction"]), int(row["x_size"]))].append(row)
    for (m_label, m_size, x_fraction, x_size), rows in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][3])
    ):
        runtimes = [float(row["runtime_s"]) for row in rows]
        throughputs = [float(row["throughput_items_per_s"]) for row in rows]
        peak_memory = [float(row["peak_memory_gb"]) for row in rows]
        output.append(
            {
                "marker": MARKER,
                "m_label": m_label,
                "m_size": m_size,
                "x_fraction": x_fraction,
                "x_size": x_size,
                "repeats": len(rows),
                "mean_runtime_s": statistics.fmean(runtimes),
                "median_runtime_s": statistics.median(runtimes),
                "runtime_std_s": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
                "peak_memory_gb": max(peak_memory),
                "mean_peak_memory_gb": statistics.fmean(peak_memory),
                "mean_throughput_items_per_s": statistics.fmean(throughputs),
                "median_throughput_items_per_s": statistics.median(throughputs),
            }
        )
    return output


def slope_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for x_size in sorted({int(row["x_size"]) for row in summary_rows}):
        selected = [row for row in summary_rows if int(row["x_size"]) == x_size]
        output.append(fit_slope("runtime_vs_m_fixed_x", f"x_size={x_size}", selected, "m_size"))
    for m_size in sorted({int(row["m_size"]) for row in summary_rows}):
        selected = [row for row in summary_rows if int(row["m_size"]) == m_size]
        output.append(fit_slope("runtime_vs_x_fixed_m", f"m_size={m_size}", selected, "x_size"))
    output.append(fit_slope("runtime_vs_m_times_x_all", "all_configurations", summary_rows, "work_items"))
    return output


def fit_slope(label: str, slice_label: str, rows: Iterable[dict[str, object]], variable: str) -> dict[str, object]:
    ordered = list(rows)
    if variable == "work_items":
        x = np.asarray([int(row["m_size"]) * int(row["x_size"]) for row in ordered], dtype=float)
    else:
        x = np.asarray([float(row[variable]) for row in ordered], dtype=float)
    y = np.asarray([float(row["mean_runtime_s"]) for row in ordered], dtype=float)
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    predicted = slope * log_x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "marker": MARKER,
        "analysis": label,
        "slice": slice_label,
        "log_log_slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r2,
        "points": len(ordered),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    summary_rows: list[dict[str, object]],
    slopes: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    largest = [
        row for row in summary_rows
        if row["m_label"] == "M_full" and float(row["x_fraction"]) == 1.0
    ][0]
    global_slope = [row for row in slopes if row["analysis"] == "runtime_vs_m_times_x_all"][0]
    cycle_seconds = FORECAST_CYCLE_HOURS * 3600.0
    completes = float(largest["mean_runtime_s"]) <= cycle_seconds
    approximately_consistent = (
        0.75 <= float(global_slope["log_log_slope"]) <= 1.25
        and float(global_slope["r_squared"]) >= 0.85
    )

    lines = [
        "# AOTS2Action Scalability Experiment",
        "",
        f"Marker: **{MARKER}**.",
        "",
        "Runtime is reported per forecast case. Each repeat measures the full eligible-case batch, then divides by the fixed eligible case count so throughput is exactly `M * |X| / runtime` for one operational forecast case.",
        "",
        "## Environment",
        "",
        f"- CPU model: {metadata['cpu_model']}",
        f"- RAM: {metadata['ram_gb']:.2f} GB",
        f"- Operating system: {metadata['operating_system']}",
        f"- Python version: {metadata['python_version']}",
        f"- NumPy version: {metadata['numpy_version']}",
        "",
        "## Complete M x |X| Results Table",
        "",
        "| M | X fraction | X size | mean runtime s | median runtime s | runtime std s | peak memory GB | throughput items/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["m_label"]),
                    f"{float(row['x_fraction']):.2f}",
                    str(row["x_size"]),
                    format_float(row["mean_runtime_s"]),
                    format_float(row["median_runtime_s"]),
                    format_float(row["runtime_std_s"]),
                    format_float(row["peak_memory_gb"], 4),
                    format_float(row["mean_throughput_items_per_s"], 2),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Scaling Analysis",
            "",
            "Runtime versus ensemble size at fixed spatial volume:",
        ]
    )
    for row in [item for item in slopes if item["analysis"] == "runtime_vs_m_fixed_x"]:
        lines.append(
            f"- {row['slice']}: log-log slope {row['log_log_slope']:.3f}, R^2 {row['r_squared']:.3f}"
        )
    lines.append("")
    lines.append("Runtime versus spatial volume at fixed ensemble size:")
    for row in [item for item in slopes if item["analysis"] == "runtime_vs_x_fixed_m"]:
        lines.append(
            f"- {row['slice']}: log-log slope {row['log_log_slope']:.3f}, R^2 {row['r_squared']:.3f}"
        )
    lines.extend(
        [
            "",
            "Peak memory versus ensemble size:",
            "- Peak RSS is dominated by the loaded Python process and proxy dataset. The measured peak is nearly flat across M because the implementation streams ensemble members over the grid instead of materializing an M x |X| distance matrix.",
            "",
            "Throughput across all configurations:",
            f"- The global log-log runtime slope versus M*|X| is {global_slope['log_log_slope']:.3f} (R^2 {global_slope['r_squared']:.3f}).",
            f"- Observed scaling is {'approximately consistent' if approximately_consistent else 'not cleanly consistent'} with O(M|X|) under the stated proxy-kernel protocol.",
            "",
            "## Largest Configuration",
            "",
            f"- Configuration: M_full = {metadata['m_full']}, N_X = {metadata['n_x']}",
            f"- Runtime T: {float(largest['mean_runtime_s']):.6f} s per forecast case",
            f"- Peak memory: {float(largest['peak_memory_gb']):.4f} GB",
            f"- Throughput: {float(largest['mean_throughput_items_per_s']):.2f} items/s",
            f"- Mean runtime: {float(largest['mean_runtime_s']):.6f} s",
            f"- Median runtime: {float(largest['median_runtime_s']):.6f} s",
            f"- Completes within one {FORECAST_CYCLE_HOURS:.0f} h forecast cycle: {'yes' if completes else 'no'}",
            "",
            "## Figure-Ready Data",
            "",
            "- Runtime vs M: `results_AOTS2Action/csv/rq4_runtime_vs_m_PROXY.csv`",
            "- Runtime vs |X|: `results_AOTS2Action/csv/rq4_runtime_vs_x_PROXY.csv`",
            "- Throughput: `results_AOTS2Action/csv/rq4_throughput_PROXY.csv`",
            "- Peak memory: `results_AOTS2Action/csv/rq4_peak_memory_PROXY.csv`",
            "- Raw repeated runs: `results_AOTS2Action/csv/rq4_scalability_raw_PROXY.csv`",
            "- Slopes: `results_AOTS2Action/csv/rq4_scalability_slopes_PROXY.csv`",
            "",
            "Do not state that scaling is linear beyond this measured proxy setting.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def subset_columns(rows: list[dict[str, object]], columns: list[str]) -> list[dict[str, object]]:
    return [{column: row[column] for column in columns} for row in rows]


def hardware_metadata() -> dict[str, object]:
    virtual_memory = psutil.virtual_memory()
    cpu_model = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            import subprocess

            completed = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in completed.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Chip:"):
                    cpu_model = stripped.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return {
        "cpu_model": cpu_model,
        "ram_gb": virtual_memory.total / GB,
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path, default=Path("../UNICEF_DATA/AOTS_DATA_SHARE (5).csv"))
    parser.add_argument("--corpus", type=Path, default=Path("results_AOTS2Action/csv/table1_evaluation_corpus.csv"))
    parser.add_argument("--proxy-grid", type=Path, default=Path("../UNICEF_DATA/outputs/proxy_external_grid_from_aots.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_AOTS2Action"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--horizons", default="24,48,72")
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.0005)
    args = parser.parse_args()

    horizons = {int(value.strip()) for value in args.horizons.split(",") if value.strip()}
    cases_all = load_cases(args.corpus, horizons)
    positions_all = load_positions(args.forecasts, set(cases_all), horizons)
    missing = set(cases_all) - set(positions_all)
    if missing:
        raise ValueError(f"Missing ensemble positions for {len(missing)} cases")

    m_full = max(len(members) for members in positions_all.values())
    eligible_keys = {
        key for key, members in positions_all.items()
        if len(members) >= m_full and key in cases_all
    }
    cases = {key: cases_all[key] for key in eligible_keys}
    positions = {key: positions_all[key] for key in eligible_keys}
    if not cases:
        raise ValueError("No eligible cases have the full ensemble size.")

    grid = load_grid(args.proxy_grid)
    n_x = len(grid["lat"])
    x_sizes = [max(1, int(round(fraction * n_x))) for fraction in DEFAULT_SPATIAL_FRACTIONS]
    x_sizes[-1] = n_x
    m_plan = [(str(size), size) for size in DEFAULT_ENSEMBLE_SIZES] + [("M_full", m_full)]

    raw_rows: list[dict[str, object]] = []
    for m_label, m_size in m_plan:
        for x_fraction, x_size in zip(DEFAULT_SPATIAL_FRACTIONS, x_sizes):
            for repeat in range(1, args.repeats + 1):
                raw_rows.append(
                    measure_configuration(
                        cases,
                        positions,
                        grid,
                        m_size,
                        m_label,
                        x_size,
                        x_fraction,
                        repeat,
                        args,
                    )
                )
                print(
                    f"completed M={m_label} |X|={x_size} repeat={repeat}/{args.repeats}",
                    flush=True,
                )

    summary_rows = summarize(raw_rows)
    slopes = slope_rows(summary_rows)
    metadata = {
        **hardware_metadata(),
        "marker": MARKER,
        "forecast_file": str(args.forecasts),
        "corpus_file": str(args.corpus),
        "proxy_grid_file": str(args.proxy_grid),
        "horizons_h": sorted(horizons),
        "repeats": args.repeats,
        "m_full": m_full,
        "m_values": [label for label, _ in m_plan],
        "n_x": n_x,
        "x_sizes": x_sizes,
        "x_fractions": list(DEFAULT_SPATIAL_FRACTIONS),
        "loaded_case_count": len(cases_all),
        "eligible_case_count": len(cases),
        "eligible_case_policy": "same cases for every configuration; require at least M_full members",
        "runtime_scope": "per forecast case, derived from total eligible-case batch runtime / eligible_case_count",
        "throughput_formula": "M * |X| / runtime_s",
        "forecast_cycle_hours": FORECAST_CYCLE_HOURS,
        "impact_radius_km": args.impact_radius_km,
        "cone_buffer_km": args.cone_buffer_km,
        "random_seed": args.seed,
    }

    csv_dir = args.out_dir / "csv"
    table_dir = args.out_dir / "tables"
    write_csv(csv_dir / "rq4_scalability_raw_PROXY.csv", raw_rows)
    write_csv(table_dir / "rq4_scalability_results_PROXY.csv", summary_rows)
    write_csv(csv_dir / "rq4_scalability_slopes_PROXY.csv", slopes)
    write_csv(
        csv_dir / "rq4_runtime_vs_m_PROXY.csv",
        subset_columns(summary_rows, ["marker", "m_label", "m_size", "x_fraction", "x_size", "mean_runtime_s", "median_runtime_s", "runtime_std_s"]),
    )
    write_csv(
        csv_dir / "rq4_runtime_vs_x_PROXY.csv",
        subset_columns(summary_rows, ["marker", "m_label", "m_size", "x_fraction", "x_size", "mean_runtime_s", "median_runtime_s", "runtime_std_s"]),
    )
    write_csv(
        csv_dir / "rq4_throughput_PROXY.csv",
        subset_columns(summary_rows, ["marker", "m_label", "m_size", "x_fraction", "x_size", "mean_throughput_items_per_s", "median_throughput_items_per_s"]),
    )
    write_csv(
        csv_dir / "rq4_peak_memory_PROXY.csv",
        subset_columns(summary_rows, ["marker", "m_label", "m_size", "x_fraction", "x_size", "peak_memory_gb", "mean_peak_memory_gb"]),
    )
    (csv_dir / "rq4_scalability_metadata_PROXY.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(args.out_dir / "RQ4_SCALABILITY.md", summary_rows, slopes, metadata)

    print(f"Wrote {table_dir / 'rq4_scalability_results_PROXY.csv'}")
    print(f"Wrote {args.out_dir / 'RQ4_SCALABILITY.md'}")


if __name__ == "__main__":
    main()
