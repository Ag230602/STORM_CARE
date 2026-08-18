#!/usr/bin/env python3
"""Build a real RQ2 grid from WorldPop population and World Bank income groups.

This is tailored for the AOTS2Action all-horizon RQ2 rerun. It keeps the
existing cyclone forecast grid, queries WorldPop API v2 for cells that can
contribute to exposure metrics, and assigns a country-level socioeconomic
vulnerability weight from World Bank income classifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

import build_aots2action_rq2 as rq2


COUNTRIES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)
WORLD_BANK_COUNTRY_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
WORLDPOP_IMAGESERVER_STATS_URL = (
    "https://worldpop.arcgis.com/arcgis/rest/services/"
    "WorldPop_Total_Population_1km/ImageServer/computeStatisticsHistograms"
)
INCOME_VULNERABILITY = {
    "LIC": 1.00,
    "LMC": 0.75,
    "UMC": 0.50,
    "HIC": 0.25,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def polygon_contains(ring: list[list[float]], lon: float, lat: float) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def geometry_contains(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        polygons = [coordinates]
    elif geometry.get("type") == "MultiPolygon":
        polygons = coordinates
    else:
        return False
    for polygon in polygons:
        if not polygon:
            continue
        if polygon_contains(polygon[0], lon, lat) and not any(
            polygon_contains(hole, lon, lat) for hole in polygon[1:]
        ):
            return True
    return False


def coordinate_in_bbox(coordinate: list[float], bbox: tuple[float, float, float, float]) -> bool:
    lon, lat = float(coordinate[0]), float(coordinate[1])
    xmin, ymin, xmax, ymax = bbox
    return xmin <= lon <= xmax and ymin <= lat <= ymax


def geometry_intersects_bbox(
    geometry: dict[str, Any], bbox: tuple[float, float, float, float]
) -> bool:
    xmin, ymin, xmax, ymax = bbox
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    if any(geometry_contains(geometry, lon, lat) for lon, lat in corners):
        return True

    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geometry.get("type") == "Polygon" else coordinates
    for polygon in polygons:
        for ring in polygon:
            if any(coordinate_in_bbox(coordinate, bbox) for coordinate in ring):
                return True
    return False


def feature_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox" in feature:
        xmin, ymin, xmax, ymax = feature["bbox"]
        return float(xmin), float(ymin), float(xmax), float(ymax)
    xs: list[float] = []
    ys: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            xs.append(float(value[0]))
            ys.append(float(value[1]))
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(feature["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def load_country_features(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(COUNTRIES_GEOJSON_URL, timeout=120)
        response.raise_for_status()
        cache_path.write_text(response.text)
    data = json.loads(cache_path.read_text())
    features = []
    for feature in data["features"]:
        props = feature.get("properties", {})
        bbox = feature_bbox(feature)
        features.append(
            {
                "bbox": bbox,
                "geometry": feature["geometry"],
                "iso3": props.get("ISO_A3") or props.get("iso_a3") or props.get("ISO3166-1-Alpha-3"),
                "name": props.get("ADMIN") or props.get("name") or props.get("NAME"),
            }
        )
    return features


def load_world_bank_income(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(WORLD_BANK_COUNTRY_URL, timeout=120)
        response.raise_for_status()
        cache_path.write_text(response.text)
    data = json.loads(cache_path.read_text())
    countries = data[1] if isinstance(data, list) and len(data) > 1 else []
    output: dict[str, dict[str, object]] = {}
    for row in countries:
        iso3 = row.get("id")
        income_id = (row.get("incomeLevel") or {}).get("id")
        output[iso3] = {
            "name": row.get("name"),
            "income_id": income_id,
            "income_value": (row.get("incomeLevel") or {}).get("value"),
            "vulnerability": INCOME_VULNERABILITY.get(income_id, 0.50),
        }
    return output


def assign_country(
    lon: float, lat: float, countries: list[dict[str, Any]], resolution_deg: float
) -> tuple[str, str]:
    for feature in countries:
        xmin, ymin, xmax, ymax = feature["bbox"]
        if xmin <= lon <= xmax and ymin <= lat <= ymax:
            if geometry_contains(feature["geometry"], lon, lat):
                return str(feature["iso3"]), str(feature["name"])
    half = resolution_deg / 2.0
    cell_bbox = (lon - half, lat - half, lon + half, lat + half)
    for feature in countries:
        xmin, ymin, xmax, ymax = feature["bbox"]
        overlaps = not (
            xmax < cell_bbox[0]
            or xmin > cell_bbox[2]
            or ymax < cell_bbox[1]
            or ymin > cell_bbox[3]
        )
        if overlaps and geometry_intersects_bbox(feature["geometry"], cell_bbox):
            return str(feature["iso3"]), str(feature["name"])
    return "UNASSIGNED", "Unassigned/ocean"


def exposure_active_cells(
    forecasts: Path,
    corpus: Path,
    grid_path: Path,
    horizons: tuple[int, ...],
    impact_radius_km: float,
    cone_buffer_km: float,
) -> set[tuple[float, float]]:
    rq2.HORIZONS = horizons
    cases = rq2.load_cases(corpus)
    positions = rq2.load_positions(forecasts, set(cases))
    grid = rq2.load_grid(grid_path)
    used = np.zeros(len(grid["lat"]), dtype=bool)
    for key, case in cases.items():
        members = np.asarray(positions[key], dtype=float)
        mean_lat = float(members[:, 0].mean())
        mean_lon = float(members[:, 1].mean())
        spread = np.asarray(
            [rq2.great_circle_km(mean_lat, mean_lon, lat, lon) for lat, lon in members]
        )
        p90_radius = float(np.quantile(spread, 0.9)) + cone_buffer_km
        used |= (
            rq2.distances_km(
                grid["lat"], grid["lon"], float(case["observed_lat"]), float(case["observed_lon"])
            )
            <= impact_radius_km
        )
        used |= rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon) <= impact_radius_km
        used |= rq2.distances_km(grid["lat"], grid["lon"], mean_lat, mean_lon) <= (
            p90_radius + impact_radius_km
        )
        for member_lat, member_lon in members:
            used |= (
                rq2.distances_km(grid["lat"], grid["lon"], member_lat, member_lon)
                <= impact_radius_km
            )
    return {
        (round(float(lat), 8), round(float(lon), 8))
        for lat, lon, active in zip(grid["lat"], grid["lon"], used)
        if active
    }


def submit_worldpop(
    cell: tuple[float, float],
    resolution_deg: float,
    year: int,
    max_retries: int = 8,
) -> str:
    lat, lon = cell
    half = resolution_deg / 2.0
    coordinates = [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]
    payload = {
        "geojson": {"type": "Polygon", "coordinates": [coordinates]},
        "year": year,
        "resolution": "1km",
    }
    response = None
    for attempt in range(max_retries):
        response = requests.post("https://api.worldpop.org/v2/population", json=payload, timeout=120)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(300.0, 30.0 * (attempt + 1))
        print(f"WorldPop 429; sleeping {delay:.0f}s before retry", flush=True)
        time.sleep(delay)
    assert response is not None
    response.raise_for_status()
    data = response.json()
    if "result" in data and data["result"]:
        return json.dumps(data["result"])
    return str(data["task_id"])


def poll_worldpop(task_id_or_result: str, max_wait_seconds: int) -> float:
    if task_id_or_result.startswith("{"):
        return float(json.loads(task_id_or_result)["total_population"])
    deadline = time.time() + max_wait_seconds
    url = f"https://api.worldpop.org/v2/tasks/{task_id_or_result}"
    while time.time() < deadline:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return float(data["result"]["total_population"])
        if data.get("status") in {"failed", "error"}:
            raise RuntimeError(f"WorldPop task failed: {data}")
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for WorldPop task {task_id_or_result}")


def load_population_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {key: float(value) for key, value in json.loads(path.read_text()).items()}


def save_population_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def cache_key(cell: tuple[float, float]) -> str:
    return f"{cell[0]:.8f},{cell[1]:.8f}"


def fetch_population_for_cells(
    cells: set[tuple[float, float]],
    cache_path: Path,
    resolution_deg: float,
    year: int,
    workers: int,
    max_wait_seconds: int,
) -> dict[str, float]:
    cache = load_population_cache(cache_path)
    lock = threading.Lock()
    missing = [cell for cell in sorted(cells) if cache_key(cell) not in cache]

    def worker(cell: tuple[float, float]) -> tuple[str, float]:
        key = cache_key(cell)
        task_id = submit_worldpop(cell, resolution_deg, year)
        population = poll_worldpop(task_id, max_wait_seconds)
        return key, population

    if missing:
        print(f"WorldPop cells to query: {len(missing)}; cached: {len(cache)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, cell): cell for cell in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            key, population = future.result()
            with lock:
                cache[key] = population
                if index % 10 == 0 or index == len(missing):
                    save_population_cache(cache_path, cache)
                    print(f"Saved {index}/{len(missing)} new WorldPop cells", flush=True)
    save_population_cache(cache_path, cache)
    return cache


def fetch_arcgis_population_for_cells(
    cells: set[tuple[float, float]],
    cache_path: Path,
    resolution_deg: float,
    year: int,
    workers: int,
) -> dict[str, float]:
    cache = load_population_cache(cache_path)
    missing = [cell for cell in sorted(cells) if cache_key(cell) not in cache]
    year_time = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    lock = threading.Lock()

    def worker(cell: tuple[float, float]) -> tuple[str, float]:
        lat, lon = cell
        half = resolution_deg / 2.0
        geometry = {
            "xmin": lon - half,
            "ymin": lat - half,
            "xmax": lon + half,
            "ymax": lat + half,
            "spatialReference": {"wkid": 4326},
        }
        params = {
            "f": "json",
            "geometry": json.dumps(geometry),
            "geometryType": "esriGeometryEnvelope",
            "time": str(year_time),
            "pixelSize": "0.0083333333,0.0083333333",
        }
        response = requests.get(WORLDPOP_IMAGESERVER_STATS_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"WorldPop ImageServer error for {cell}: {data['error']}")
        statistics = data.get("statistics") or []
        population = float((statistics[0] or {}).get("sum") or 0.0) if statistics else 0.0
        return cache_key(cell), population

    if missing:
        print(f"WorldPop ImageServer cells to query: {len(missing)}; cached: {len(cache)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, cell): cell for cell in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            key, population = future.result()
            with lock:
                cache[key] = population
                if index % 25 == 0 or index == len(missing):
                    save_population_cache(cache_path, cache)
                    print(f"Saved {index}/{len(missing)} new ImageServer cells", flush=True)
    save_population_cache(cache_path, cache)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-grid", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/real_rq2"))
    parser.add_argument("--horizons", default="6,12,24,48,72,96")
    parser.add_argument("--impact-radius-km", type=float, default=25.0)
    parser.add_argument("--cone-buffer-km", type=float, default=25.0)
    parser.add_argument("--grid-resolution-deg", type=float, default=0.75)
    parser.add_argument("--worldpop-year", type=int, default=2020)
    parser.add_argument("--population-mode", choices=("imageserver", "api-v2"), default="imageserver")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-wait-seconds", type=int, default=180)
    args = parser.parse_args()

    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    grid_rows = read_rows(args.forecast_grid)
    active = exposure_active_cells(
        args.forecasts,
        args.corpus,
        args.forecast_grid,
        horizons,
        args.impact_radius_km,
        args.cone_buffer_km,
    )
    countries = load_country_features(args.cache_dir / "countries.geojson")
    income = load_world_bank_income(args.cache_dir / "world_bank_countries.json")
    if args.population_mode == "api-v2":
        population_cache = fetch_population_for_cells(
            active,
            args.cache_dir / "worldpop_population_cache.json",
            args.grid_resolution_deg,
            args.worldpop_year,
            args.workers,
            args.max_wait_seconds,
        )
        population_api = "https://api.worldpop.org/v2/population"
        population_source = "WorldPop API v2 population, R2025A 1km"
    else:
        population_cache = fetch_arcgis_population_for_cells(
            active,
            args.cache_dir / "worldpop_imageserver_population_cache.json",
            args.grid_resolution_deg,
            args.worldpop_year,
            args.workers,
        )
        population_api = WORLDPOP_IMAGESERVER_STATS_URL
        population_source = "WorldPop Total Population 1km ArcGIS ImageServer"

    output_rows: list[dict[str, object]] = []
    unassigned = 0
    for row in grid_rows:
        lat = round(float(row["lat"]), 8)
        lon = round(float(row["lon"]), 8)
        iso3, country_name = assign_country(lon, lat, countries, args.grid_resolution_deg)
        wb = income.get(iso3, {"income_id": "NA", "income_value": "Not classified", "vulnerability": 0.50})
        if iso3 == "UNASSIGNED":
            unassigned += 1
        key = cache_key((lat, lon))
        output_rows.append(
            {
                "lat": lat,
                "lon": lon,
                "population": population_cache.get(key, 0.0),
                "inform_risk": wb["vulnerability"],
                "region_id": iso3,
                "region_name": country_name,
                "country_income_group": wb["income_id"],
                "country_income_group_label": wb["income_value"],
                "rq2_exposure_active_cell": key in population_cache,
                "population_source": population_source,
                "vulnerability_source": "World Bank country income classification mapped to vulnerability weights",
                "admin_source": "Natural Earth country boundaries via datasets/geo-countries",
                "infrastructure_source": "not supplied by this builder; optional infrastructure augmentation may update this field",
            }
        )

    write_csv(args.output, output_rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(
            {
                "marker": "REAL_HUMANITARIAN_GEOSPATIAL_DATA",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "cyclone_grid_template": "AOTS-derived cyclone forecast grid harmonized to real geospatial sources",
                "output_grid": str(args.output),
                "horizons_h": list(horizons),
                "grid_cells_total": len(grid_rows),
                "rq2_exposure_active_cells": len(active),
                "worldpop_cells_cached_or_queried": len(population_cache),
                "worldpop_year": args.worldpop_year,
                "worldpop_resolution": "1km",
                "worldpop_population_mode": args.population_mode,
                "worldpop_api": population_api,
                "vulnerability": {
                    "source": "World Bank country income classification",
                    "api": WORLD_BANK_COUNTRY_URL,
                    "mapping": INCOME_VULNERABILITY,
                    "fallback_weight": 0.50,
                },
                "administrative_boundaries": {
                    "source": "Natural Earth country boundaries via geo-countries",
                    "url": COUNTRIES_GEOJSON_URL,
                    "unassigned_or_ocean_grid_cells": unassigned,
                },
                "inactive_cell_population_policy": (
                    "Cells outside every RQ2 realized, deterministic, P90, and member footprint "
                    "are set to zero population because they cannot affect exposure AE/signed/ratio; "
                    "Brier scores are unweighted over grid cells."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
