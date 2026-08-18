"""Add lightweight real infrastructure indicators to the AOTS2Action real grid.

The primary RQ2/RQ3 exposure metric remains population * vulnerability. This
script adds harmonized infrastructure context columns from public Natural Earth
transport layers so the real grid retains infrastructure source information.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_AOTS2Action"
CACHE = ROOT / "data_cache" / "real_infrastructure"

SOURCES = {
    "airports": {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_airports.geojson",
        "version": "Natural Earth 1:10m airports, version 5.0.0",
    },
    "ports": {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_ports.geojson",
        "version": "Natural Earth 1:10m ports, version 5.0.0",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_json(name: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"ne_10m_{name}.geojson"
    if not path.exists():
        url = SOURCES[name]["url"]
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())


def load_points(name: str) -> list[dict[str, object]]:
    data = download_json(name)
    points = []
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom.get("coordinates", [None, None])[:2]
        if lat is None or lon is None:
            continue
        props = feature.get("properties") or {}
        points.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "name": props.get("name") or props.get("nameascii") or "",
                "scalerank": float(props.get("scalerank", 99) or 99),
            }
        )
    return points


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def nearby_counts(lat: float, lon: float, points: list[dict[str, object]], radius_km: float) -> tuple[int, float]:
    count = 0
    nearest = math.inf
    for point in points:
        dist = haversine_km(lat, lon, float(point["lat"]), float(point["lon"]))
        if dist < nearest:
            nearest = dist
        if dist <= radius_km:
            count += 1
    return count, nearest


def main() -> None:
    grid_path = RESULTS / "csv" / "humanitarian_grid_REAL.csv"
    metadata_path = RESULTS / "csv" / "humanitarian_grid_REAL.metadata.json"
    rows = read_rows(grid_path)
    airports = load_points("airports")
    ports = load_points("ports")
    radius_km = 100.0

    max_score = 0.0
    augmented = []
    for row in rows:
        lat = float(row["lat"])
        lon = float(row["lon"])
        airport_count, nearest_airport = nearby_counts(lat, lon, airports, radius_km)
        port_count, nearest_port = nearby_counts(lat, lon, ports, radius_km)
        score = airport_count + port_count
        max_score = max(max_score, score)
        row = dict(row)
        row["airport_count_100km"] = airport_count
        row["port_count_100km"] = port_count
        row["nearest_airport_km"] = "" if math.isinf(nearest_airport) else f"{nearest_airport:.3f}"
        row["nearest_port_km"] = "" if math.isinf(nearest_port) else f"{nearest_port:.3f}"
        row["infrastructure_access_score"] = score
        row["infrastructure_source"] = (
            "Natural Earth 1:10m airports v5.0.0 and ports v5.0.0; "
            "counts within 100 km of harmonized grid-cell center"
        )
        augmented.append(row)

    for row in augmented:
        raw = float(row["infrastructure_access_score"])
        row["infrastructure_access_score_norm"] = f"{(raw / max_score if max_score else 0.0):.6f}"

    fieldnames = list(augmented[0].keys())
    write_rows(grid_path, augmented, fieldnames)

    metadata = json.loads(metadata_path.read_text())
    metadata["infrastructure"] = {
        "sources": {
            key: {
                "url": value["url"],
                "version": value["version"],
            }
            for key, value in SOURCES.items()
        },
        "harmonization": (
            "Airport and port points are counted within 100 km of each cyclone-grid cell center; "
            "nearest distances are retained in km."
        ),
        "airports_loaded": len(airports),
        "ports_loaded": len(ports),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_exposure_weight_note": (
            "Infrastructure columns are retained as real spatial context/provenance; "
            "the primary RQ2/RQ3 exposure weight remains population * inform_risk."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n")
    print(f"Updated {grid_path} with {len(airports)} airports and {len(ports)} ports")
    print(f"Updated {metadata_path}")


if __name__ == "__main__":
    main()
