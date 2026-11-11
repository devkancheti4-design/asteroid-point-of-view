"""Fetch NEO orbits (SBDB Query API) and Earth close approaches (CAD API) from NASA/JPL.

Outputs
  data/raw/sbdb_neo.json          raw API response
  data/processed/sbdb_neo.csv     one row per near-Earth asteroid (orbit, size, uncertainty, label)
  data/raw/cad_<y0>_<y1>.json     raw API responses, one per time chunk
  data/processed/cad_earth.csv    one row per Earth close approach (the "Earth's view" data)
  data/processed/provenance.json  retrieval time and query filters, for the paper
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"

SBDB_FIELDS = [
    "spkid", "pdes", "full_name", "neo", "pha", "class",
    "H", "diameter", "diameter_sigma", "albedo",
    "epoch", "e", "a", "q", "i", "om", "w", "ma", "ad", "n", "tp", "per",
    "moid", "moid_jup", "t_jup",
    "condition_code", "data_arc", "first_obs", "last_obs", "n_obs_used", "rms",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om", "sigma_w", "sigma_ma",
    "sigma_tp", "sigma_n", "sigma_per", "sigma_ad",
    "two_body", "A1", "A2", "A3",
]

# Mirror the NeoWs feed behind the popular Kaggle datasets: approaches within 0.5 au, 1910-2024.
CAD_YEARS = (1910, 2024)
CAD_STEP_YEARS = 5
CAD_DIST_MAX_AU = "0.5"


def get_json(url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=600)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 5 * 2**attempt
            print(f"  request failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_sbdb() -> pd.DataFrame:
    params = {"fields": ",".join(SBDB_FIELDS), "sb-group": "neo", "sb-kind": "a", "full-prec": "true"}
    js = get_json(SBDB_URL, params)
    (RAW / "sbdb_neo.json").write_text(json.dumps(js))
    df = pd.DataFrame(js["data"], columns=js["fields"])
    df.to_csv(PROC / "sbdb_neo.csv", index=False)
    print(f"SBDB: {len(df):,} near-Earth asteroids (API count {js.get('count')})")
    return df


def fetch_cad() -> pd.DataFrame:
    frames = []
    y_start, y_end = CAD_YEARS
    for y0 in range(y_start, y_end + 1, CAD_STEP_YEARS):
        y1 = min(y0 + CAD_STEP_YEARS - 1, y_end)
        params = {
            "date-min": f"{y0}-01-01",
            "date-max": f"{y1}-12-31",
            "dist-max": CAD_DIST_MAX_AU,
            "body": "Earth",
            "neo": "true",
            "fullname": "true",
        }
        js = get_json(CAD_URL, params)
        (RAW / f"cad_{y0}_{y1}.json").write_text(json.dumps(js))
        n = int(js.get("count", 0))
        print(f"CAD {y0}-{y1}: {n:,} approaches")
        if n:
            frames.append(pd.DataFrame(js["data"], columns=js["fields"]))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(PROC / "cad_earth.csv", index=False)
    print(f"CAD total: {len(df):,} approaches of {df['des'].nunique():,} distinct objects")
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sbdb = fetch_sbdb()
    cad = fetch_cad()
    provenance = {
        "retrieved_utc": retrieved,
        "sbdb": {"url": SBDB_URL, "filters": {"sb-group": "neo", "sb-kind": "a"}, "rows": len(sbdb)},
        "cad": {
            "url": CAD_URL,
            "filters": {
                "date": f"{CAD_YEARS[0]}-01-01..{CAD_YEARS[1]}-12-31",
                "dist-max_au": float(CAD_DIST_MAX_AU),
                "body": "Earth",
                "neo": True,
            },
            "rows": len(cad),
        },
    }
    (PROC / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
