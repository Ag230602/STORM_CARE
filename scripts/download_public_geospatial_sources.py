#!/usr/bin/env python3
"""Download public humanitarian/geospatial source files for real-data RQ2/RQ3.

This helper is intentionally conservative: it records source URLs and writes a
manifest, but it does not silently select one vulnerability or population
definition for the paper. Use the downloaded files as inputs to
build_real_humanitarian_grid.py after confirming year, geography, and license.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests


CDC_SVI_2022_COUNTY_LAYER = (
    "https://onemap.cdc.gov/onemapservices/rest/services/SVI/"
    "CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/1"
)
CDC_SVI_2022_TRACT_LAYER = (
    "https://onemap.cdc.gov/onemapservices/rest/services/SVI/"
    "CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/0"
)
WORLDPOP_CATALOG_URL = "https://www.worldpop.org/rest/data/pop/wpgp"


def download_file(url: str, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as destination:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    destination.write(chunk)
    return {"url": url, "path": str(path), "bytes": path.stat().st_size}


def download_cdc_svi(layer_url: str, path: Path, page_size: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, object]] = []
    offset = 0
    while True:
        query = urlencode(
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
        )
        response = requests.get(f"{layer_url}/query?{query}", timeout=120)
        response.raise_for_status()
        data = response.json()
        page = data.get("features", [])
        features.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    feature_collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(feature_collection))
    return {"url": layer_url, "path": str(path), "features": len(features)}


def download_worldpop_catalog(iso3: str, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WORLDPOP_CATALOG_URL}?iso3={iso3.upper()}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_text(json.dumps(response.json(), indent=2) + "\n")
    return {"url": url, "path": str(path)}


def tiger_url(year: int, geography: str, state_fips: str | None) -> str:
    geography = geography.lower()
    if geography == "county":
        return (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/"
            f"tl_{year}_us_county.zip"
        )
    if geography == "tract":
        if not state_fips:
            raise ValueError("--state-fips is required for TIGER tract downloads")
        return (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
            f"tl_{year}_{state_fips}_tract.zip"
        )
    raise ValueError("--tiger-geography must be county or tract")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data_cache/public_geodata"))
    parser.add_argument("--cdc-svi", choices=("county", "tract"))
    parser.add_argument("--cdc-page-size", type=int, default=2000)
    parser.add_argument("--worldpop-iso3")
    parser.add_argument("--tiger-year", type=int)
    parser.add_argument("--tiger-geography", choices=("county", "tract"), default="county")
    parser.add_argument("--state-fips")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": [],
    }
    downloads: list[dict[str, object]] = manifest["downloads"]  # type: ignore[assignment]

    if args.cdc_svi:
        layer = CDC_SVI_2022_COUNTY_LAYER if args.cdc_svi == "county" else CDC_SVI_2022_TRACT_LAYER
        output = args.output_dir / f"cdc_svi_2022_us_{args.cdc_svi}.geojson"
        downloads.append(download_cdc_svi(layer, output, args.cdc_page_size))

    if args.worldpop_iso3:
        output = args.output_dir / f"worldpop_catalog_{args.worldpop_iso3.upper()}.json"
        downloads.append(download_worldpop_catalog(args.worldpop_iso3, output))

    if args.tiger_year:
        url = tiger_url(args.tiger_year, args.tiger_geography, args.state_fips)
        suffix = "us_county" if args.tiger_geography == "county" else f"{args.state_fips}_tract"
        output = args.output_dir / f"tl_{args.tiger_year}_{suffix}.zip"
        downloads.append(download_file(url, output))

    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
