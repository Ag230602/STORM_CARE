#!/usr/bin/env python3
"""Build a real humanitarian/geospatial grid for AOTS2Action RQ2/RQ3.

The output intentionally matches the existing RQ2/RQ3 grid contract:

    lat, lon, population, inform_risk, region_id

Additional provenance columns and a sidecar metadata JSON retain source
information for population, vulnerability, administrative boundaries, and
infrastructure layers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_POPULATION_SOURCE = "WorldPop/GHSL/user-supplied population layer"
DEFAULT_VULNERABILITY_SOURCE = "user-supplied vulnerability/socioeconomic layer"
DEFAULT_ADMIN_SOURCE = "user-supplied administrative boundaries"
DEFAULT_INFRA_SOURCE = "user-supplied infrastructure layer"


@dataclass(frozen=True)
class GridCell:
    lat: float
    lon: float


@dataclass
class VectorFeature:
    geometry: Any
    properties: dict[str, Any]


def require_shapely() -> tuple[Any, Any, Any]:
    try:
        from shapely.geometry import Point, shape
        from shapely.prepared import prep
    except ImportError as exc:
        raise ImportError(
            "Vector boundary/infrastructure harmonization requires shapely. "
            "Install project requirements before using --admin-boundaries, "
            "--vulnerability-vector, or --infrastructure-vector."
        ) from exc
    return Point, shape, prep


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def coerce_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def load_forecast_grid(path: Path) -> list[GridCell]:
    rows = read_csv_rows(path)
    cells = {
        (round(float(row["lat"]), 8), round(float(row["lon"]), 8))
        for row in rows
        if row.get("lat") not in (None, "") and row.get("lon") not in (None, "")
    }
    return [GridCell(lat, lon) for lat, lon in sorted(cells)]


def load_points_from_forecast_csv(path: Path) -> list[tuple[float, float]]:
    rows = read_csv_rows(path)
    points: list[tuple[float, float]] = []
    for row in rows:
        lat_value = row.get("LATITUDE") or row.get("lat") or row.get("latitude")
        lon_value = row.get("LONGITUDE") or row.get("lon") or row.get("longitude")
        if lat_value in (None, "") or lon_value in (None, ""):
            continue
        points.append((float(lat_value), float(lon_value)))
    return points


def load_points_from_corpus_csv(path: Path) -> list[tuple[float, float]]:
    rows = read_csv_rows(path)
    points: list[tuple[float, float]] = []
    for row in rows:
        lat_value = row.get("observed_lat") or row.get("lat") or row.get("latitude")
        lon_value = row.get("observed_lon") or row.get("lon") or row.get("longitude")
        if lat_value in (None, "") or lon_value in (None, ""):
            continue
        points.append((float(lat_value), float(lon_value)))
    return points


def build_grid_from_extent(
    points: list[tuple[float, float]], resolution_deg: float, buffer_deg: float
) -> list[GridCell]:
    if not points:
        raise ValueError("Cannot build a grid without forecast/corpus points")
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]
    lat_min = math.floor((min(lats) - buffer_deg) / resolution_deg) * resolution_deg
    lat_max = math.ceil((max(lats) + buffer_deg) / resolution_deg) * resolution_deg
    lon_min = math.floor((min(lons) - buffer_deg) / resolution_deg) * resolution_deg
    lon_max = math.ceil((max(lons) + buffer_deg) / resolution_deg) * resolution_deg
    cells: list[GridCell] = []
    lat = lat_min
    while lat <= lat_max + 1e-9:
        lon = lon_min
        while lon <= lon_max + 1e-9:
            cells.append(GridCell(round(lat, 8), round(lon, 8)))
            lon += resolution_deg
        lat += resolution_deg
    return cells


def read_vector(path: Path) -> list[VectorFeature]:
    _, shape, _ = require_shapely()
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        data = json.loads(path.read_text())
        return [
            VectorFeature(shape(feature["geometry"]), dict(feature.get("properties") or {}))
            for feature in data.get("features", [])
            if feature.get("geometry")
        ]
    if suffix == ".shp":
        import shapefile

        reader = shapefile.Reader(str(path))
        fields = [field[0] for field in reader.fields[1:]]
        features: list[VectorFeature] = []
        for record in reader.iterShapeRecords():
            properties = dict(zip(fields, record.record))
            features.append(VectorFeature(shape(record.shape.__geo_interface__), properties))
        return features
    raise ValueError(f"Unsupported vector format for {path}; use GeoJSON or .shp")


def assign_admin_regions(
    cells: list[GridCell],
    admin_path: Path | None,
    id_field: str,
    name_field: str | None,
) -> tuple[dict[GridCell, dict[str, str]], dict[str, Any]]:
    assignments: dict[GridCell, dict[str, str]] = {}
    if admin_path is None:
        for cell in cells:
            assignments[cell] = {
                "region_id": f"cell_{cell.lat:.4f}_{cell.lon:.4f}",
                "region_name": "",
            }
        return assignments, {"admin_features": 0, "missing_admin_cells": len(cells)}

    features = read_vector(admin_path)
    Point, _, prep = require_shapely()
    prepared = [(prep(feature.geometry), feature.properties) for feature in features]
    missing = 0
    for cell in cells:
        point = Point(cell.lon, cell.lat)
        matched: dict[str, str] | None = None
        for geometry, properties in prepared:
            if geometry.contains(point) or geometry.intersects(point):
                region_id = str(properties.get(id_field, "")).strip()
                region_name = str(properties.get(name_field, "")).strip() if name_field else ""
                matched = {"region_id": region_id, "region_name": region_name}
                break
        if matched is None or not matched["region_id"]:
            missing += 1
            matched = {"region_id": f"unassigned_{cell.lat:.4f}_{cell.lon:.4f}", "region_name": ""}
        assignments[cell] = matched
    return assignments, {"admin_features": len(features), "missing_admin_cells": missing}


def allocate_point_csv_to_cells(
    cells: list[GridCell],
    path: Path,
    lat_column: str,
    lon_column: str,
    value_column: str,
    reducer: str,
) -> dict[GridCell, float]:
    values: dict[GridCell, list[float]] = defaultdict(list)
    lookup = {(cell.lat, cell.lon): cell for cell in cells}
    lats = sorted({cell.lat for cell in cells})
    lons = sorted({cell.lon for cell in cells})
    if len(lats) < 2 or len(lons) < 2:
        lat_step = lon_step = 0.25
    else:
        lat_step = min(b - a for a, b in zip(lats, lats[1:]) if b > a)
        lon_step = min(b - a for a, b in zip(lons, lons[1:]) if b > a)
    for row in read_csv_rows(path):
        lat = coerce_float(row.get(lat_column), float("nan"))
        lon = coerce_float(row.get(lon_column), float("nan"))
        if math.isnan(lat) or math.isnan(lon):
            continue
        cell_lat = round(min(lats, key=lambda candidate: abs(candidate - lat)), 8)
        cell_lon = round(min(lons, key=lambda candidate: abs(candidate - lon)), 8)
        if abs(cell_lat - lat) > lat_step / 2 + 1e-9 or abs(cell_lon - lon) > lon_step / 2 + 1e-9:
            continue
        values[lookup[(cell_lat, cell_lon)]].append(coerce_float(row.get(value_column)))

    reduced: dict[GridCell, float] = {}
    for cell in cells:
        cell_values = values.get(cell, [])
        if not cell_values:
            reduced[cell] = 0.0 if reducer == "sum" else 1.0
        elif reducer == "mean":
            reduced[cell] = float(sum(cell_values) / len(cell_values))
        else:
            reduced[cell] = float(sum(cell_values))
    return reduced


def load_population(
    cells: list[GridCell],
    population_csv: Path | None,
    args: argparse.Namespace,
) -> tuple[dict[GridCell, float], dict[str, Any]]:
    if population_csv is not None:
        values = allocate_point_csv_to_cells(
            cells,
            population_csv,
            args.population_lat_column,
            args.population_lon_column,
            args.population_value_column,
            "sum",
        )
        return values, {
            "population_mode": "point_csv_nearest_cell_sum",
            "population_rows": len(read_csv_rows(population_csv)),
        }

    if not args.worldpop_api:
        if args.allow_missing_population:
            return {cell: 0.0 for cell in cells}, {"population_mode": "missing_zero_filled"}
        raise ValueError("Provide --population-csv or --worldpop-api for real population data")

    values: dict[GridCell, float] = {}
    for index, cell in enumerate(cells, start=1):
        values[cell] = query_worldpop_cell(
            cell, args.grid_resolution_deg, args.worldpop_year, args.worldpop_resolution
        )
        if args.worldpop_sleep_seconds > 0 and index < len(cells):
            time.sleep(args.worldpop_sleep_seconds)
    return values, {
        "population_mode": "worldpop_api_v2_population",
        "worldpop_year": args.worldpop_year,
        "worldpop_resolution": args.worldpop_resolution,
        "worldpop_queries": len(cells),
    }


def query_worldpop_cell(
    cell: GridCell, resolution_deg: float, year: int, worldpop_resolution: str
) -> float:
    half = resolution_deg / 2.0
    coordinates = [
        [cell.lon - half, cell.lat - half],
        [cell.lon + half, cell.lat - half],
        [cell.lon + half, cell.lat + half],
        [cell.lon - half, cell.lat + half],
        [cell.lon - half, cell.lat - half],
    ]
    payload = {
        "geojson": {"type": "Polygon", "coordinates": [coordinates]},
        "year": year,
        "resolution": worldpop_resolution,
    }
    response = requests.post("https://api.worldpop.org/v2/population", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    for key in ("population", "total_population", "pop"):
        if key in data:
            return coerce_float(data[key])
    if isinstance(data.get("data"), dict):
        for key in ("population", "total_population", "pop"):
            if key in data["data"]:
                return coerce_float(data["data"][key])
    raise ValueError(f"WorldPop response did not contain a population total: {data}")


def load_vulnerability(
    cells: list[GridCell],
    admin_assignments: dict[GridCell, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[GridCell, float], dict[str, Any]]:
    if args.vulnerability_csv is not None:
        rows = read_csv_rows(args.vulnerability_csv)
        if args.vulnerability_lat_column and args.vulnerability_lon_column:
            values = allocate_point_csv_to_cells(
                cells,
                args.vulnerability_csv,
                args.vulnerability_lat_column,
                args.vulnerability_lon_column,
                args.vulnerability_value_column,
                "mean",
            )
            return values, {
                "vulnerability_mode": "point_csv_nearest_cell_mean",
                "vulnerability_rows": len(rows),
            }
        by_region = {
            str(row[args.vulnerability_id_column]).strip(): coerce_float(
                row.get(args.vulnerability_value_column), args.default_vulnerability
            )
            for row in rows
            if row.get(args.vulnerability_id_column) not in (None, "")
        }
        return {
            cell: by_region.get(
                admin_assignments[cell]["region_id"], args.default_vulnerability
            )
            for cell in cells
        }, {"vulnerability_mode": "region_csv_join", "vulnerability_rows": len(rows)}

    if args.vulnerability_vector is not None:
        features = read_vector(args.vulnerability_vector)
        Point, _, prep = require_shapely()
        prepared = [(prep(feature.geometry), feature.properties) for feature in features]
        values: dict[GridCell, float] = {}
        for cell in cells:
            point = Point(cell.lon, cell.lat)
            value = args.default_vulnerability
            for geometry, properties in prepared:
                if geometry.contains(point) or geometry.intersects(point):
                    value = coerce_float(
                        properties.get(args.vulnerability_value_field),
                        args.default_vulnerability,
                    )
                    break
            values[cell] = value
        return values, {
            "vulnerability_mode": "vector_point_in_polygon",
            "vulnerability_features": len(features),
        }

    return {
        cell: args.default_vulnerability for cell in cells
    }, {"vulnerability_mode": "default_constant", "default_vulnerability": args.default_vulnerability}


def load_infrastructure(
    cells: list[GridCell], args: argparse.Namespace
) -> tuple[dict[GridCell, Counter[str]], dict[str, Any]]:
    counters: dict[GridCell, Counter[str]] = {cell: Counter() for cell in cells}
    if args.infrastructure_csv is None and args.infrastructure_vector is None:
        return counters, {"infrastructure_mode": "not_supplied"}

    lats = sorted({cell.lat for cell in cells})
    lons = sorted({cell.lon for cell in cells})
    lookup = {(cell.lat, cell.lon): cell for cell in cells}

    def add_point(lat: float, lon: float, feature_type: str) -> None:
        cell_lat = round(min(lats, key=lambda candidate: abs(candidate - lat)), 8)
        cell_lon = round(min(lons, key=lambda candidate: abs(candidate - lon)), 8)
        counters[lookup[(cell_lat, cell_lon)]][feature_type or "infrastructure"] += 1

    if args.infrastructure_csv is not None:
        rows = read_csv_rows(args.infrastructure_csv)
        for row in rows:
            lat = coerce_float(row.get(args.infrastructure_lat_column), float("nan"))
            lon = coerce_float(row.get(args.infrastructure_lon_column), float("nan"))
            if math.isnan(lat) or math.isnan(lon):
                continue
            add_point(lat, lon, str(row.get(args.infrastructure_type_column, "infrastructure")))
        return counters, {"infrastructure_mode": "point_csv_count", "infrastructure_rows": len(rows)}

    features = read_vector(args.infrastructure_vector)
    for feature in features:
        centroid = feature.geometry.centroid
        add_point(
            float(centroid.y),
            float(centroid.x),
            str(feature.properties.get(args.infrastructure_type_field, "infrastructure")),
        )
    return counters, {
        "infrastructure_mode": "vector_centroid_count",
        "infrastructure_features": len(features),
    }


def write_grid(
    path: Path,
    cells: list[GridCell],
    admin: dict[GridCell, dict[str, str]],
    population: dict[GridCell, float],
    vulnerability: dict[GridCell, float],
    infrastructure: dict[GridCell, Counter[str]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lat",
        "lon",
        "population",
        "inform_risk",
        "region_id",
        "region_name",
        "infrastructure_count",
        "infrastructure_types_json",
        "population_source",
        "vulnerability_source",
        "admin_source",
        "infrastructure_source",
    ]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            infra_counts = infrastructure[cell]
            writer.writerow(
                {
                    "lat": cell.lat,
                    "lon": cell.lon,
                    "population": population[cell],
                    "inform_risk": vulnerability[cell],
                    "region_id": admin[cell]["region_id"],
                    "region_name": admin[cell]["region_name"],
                    "infrastructure_count": sum(infra_counts.values()),
                    "infrastructure_types_json": json.dumps(dict(infra_counts), sort_keys=True),
                    "population_source": args.population_source_name,
                    "vulnerability_source": args.vulnerability_source_name,
                    "admin_source": args.admin_source_name,
                    "infrastructure_source": args.infrastructure_source_name,
                }
            )


def build_metadata(
    cells: list[GridCell],
    admin_stats: dict[str, Any],
    population_stats: dict[str, Any],
    vulnerability_stats: dict[str, Any],
    infrastructure_stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "marker": "REAL_HUMANITARIAN_GEOSPATIAL_DATA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_grid": str(args.output),
        "grid": {
            "cell_count": len(cells),
            "grid_source": str(args.forecast_grid) if args.forecast_grid else "forecast/corpus extent",
            "resolution_deg": args.grid_resolution_deg,
            "buffer_deg": args.grid_buffer_deg,
            "contract": ["lat", "lon", "population", "inform_risk", "region_id"],
        },
        "sources": {
            "population": {
                "name": args.population_source_name,
                "path": str(args.population_csv) if args.population_csv else None,
                "url": args.population_source_url,
                **population_stats,
            },
            "vulnerability": {
                "name": args.vulnerability_source_name,
                "path": str(args.vulnerability_csv or args.vulnerability_vector)
                if (args.vulnerability_csv or args.vulnerability_vector)
                else None,
                "url": args.vulnerability_source_url,
                **vulnerability_stats,
            },
            "administrative_boundaries": {
                "name": args.admin_source_name,
                "path": str(args.admin_boundaries) if args.admin_boundaries else None,
                "url": args.admin_source_url,
                **admin_stats,
            },
            "infrastructure": {
                "name": args.infrastructure_source_name,
                "path": str(args.infrastructure_csv or args.infrastructure_vector)
                if (args.infrastructure_csv or args.infrastructure_vector)
                else None,
                "url": args.infrastructure_source_url,
                **infrastructure_stats,
            },
        },
        "harmonization": {
            "population": "point totals are summed to nearest cyclone grid cell, or WorldPop API totals are requested per cell polygon",
            "vulnerability": "region joins use administrative region_id; vector joins use cell-centroid point-in-polygon; point layers use nearest-cell mean",
            "administrative_boundaries": "cell centroids are assigned to containing/intersecting administrative polygons",
            "infrastructure": "point or vector-centroid features are counted by nearest cyclone grid cell",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-grid", type=Path)
    parser.add_argument("--forecasts", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--grid-resolution-deg", type=float, default=0.25)
    parser.add_argument("--grid-buffer-deg", type=float, default=2.0)

    parser.add_argument("--population-csv", type=Path)
    parser.add_argument("--population-lat-column", default="lat")
    parser.add_argument("--population-lon-column", default="lon")
    parser.add_argument("--population-value-column", default="population")
    parser.add_argument("--population-source-name", default=DEFAULT_POPULATION_SOURCE)
    parser.add_argument("--population-source-url")
    parser.add_argument("--worldpop-api", action="store_true")
    parser.add_argument("--worldpop-year", type=int, default=2020)
    parser.add_argument("--worldpop-resolution", default="100m")
    parser.add_argument("--worldpop-sleep-seconds", type=float, default=0.2)
    parser.add_argument("--allow-missing-population", action="store_true")

    parser.add_argument("--vulnerability-csv", type=Path)
    parser.add_argument("--vulnerability-vector", type=Path)
    parser.add_argument("--vulnerability-id-column", default="region_id")
    parser.add_argument("--vulnerability-value-column", default="inform_risk")
    parser.add_argument("--vulnerability-value-field", default="RPL_THEMES")
    parser.add_argument("--vulnerability-lat-column")
    parser.add_argument("--vulnerability-lon-column")
    parser.add_argument("--vulnerability-source-name", default=DEFAULT_VULNERABILITY_SOURCE)
    parser.add_argument("--vulnerability-source-url")
    parser.add_argument("--default-vulnerability", type=float, default=1.0)

    parser.add_argument("--admin-boundaries", type=Path)
    parser.add_argument("--admin-id-field", default="GEOID")
    parser.add_argument("--admin-name-field", default="NAME")
    parser.add_argument("--admin-source-name", default=DEFAULT_ADMIN_SOURCE)
    parser.add_argument("--admin-source-url")

    parser.add_argument("--infrastructure-csv", type=Path)
    parser.add_argument("--infrastructure-vector", type=Path)
    parser.add_argument("--infrastructure-lat-column", default="lat")
    parser.add_argument("--infrastructure-lon-column", default="lon")
    parser.add_argument("--infrastructure-type-column", default="type")
    parser.add_argument("--infrastructure-type-field", default="type")
    parser.add_argument("--infrastructure-source-name", default=DEFAULT_INFRA_SOURCE)
    parser.add_argument("--infrastructure-source-url")
    args = parser.parse_args()

    if args.forecast_grid is not None:
        cells = load_forecast_grid(args.forecast_grid)
    else:
        points: list[tuple[float, float]] = []
        if args.forecasts is not None:
            points.extend(load_points_from_forecast_csv(args.forecasts))
        if args.corpus is not None:
            points.extend(load_points_from_corpus_csv(args.corpus))
        cells = build_grid_from_extent(points, args.grid_resolution_deg, args.grid_buffer_deg)

    admin_assignments, admin_stats = assign_admin_regions(
        cells, args.admin_boundaries, args.admin_id_field, args.admin_name_field
    )
    population, population_stats = load_population(cells, args.population_csv, args)
    vulnerability, vulnerability_stats = load_vulnerability(cells, admin_assignments, args)
    infrastructure, infrastructure_stats = load_infrastructure(cells, args)

    write_grid(
        args.output,
        cells,
        admin_assignments,
        population,
        vulnerability,
        infrastructure,
        args,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(
            build_metadata(
                cells,
                admin_stats,
                population_stats,
                vulnerability_stats,
                infrastructure_stats,
                args,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
