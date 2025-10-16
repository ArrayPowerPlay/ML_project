#!/usr/bin/env python3
"""
build_worldbank_dataset.py

Fetches selected World Bank indicators for ALL countries, years 2000–2024,
and writes a single-sheet Excel file with columns:
Country Name | Country Code | Year | <12 indicators in English>

Usage (from repo root):
  python scripts/build_worldbank_dataset.py --out data/worldbank_2000_2024.xlsx

Optional args:
  --start 2000 --end 2024
  --csv data/worldbank_2000_2024.csv          # also save CSV
  --include-aggregates                        # keep regions/aggregates (default: exclude)
"""

import argparse
import sys
import time
import requests
import pandas as pd
from typing import Dict, List

DEFAULT_START = 2000
DEFAULT_END = 2024

# Mapping of indicator codes -> column label (exact English names from World Bank)
INDICATORS = {
    # Population
    "SP.POP.TOTL": "Population, total",
    # Poverty
    "SI.POV.DDAY": "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)",
    # Demographics
    "SP.POP.GROW": "Population growth (annual %)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    # Economy
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    # Services / Access
    "SH.STA.SMSS.ZS": "People using safely managed sanitation services (% of population)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
    "SH.H2O.BASW.ZS": "People using at least basic drinking water services (% of population)",
    # Environment / Urban / Labor
    "EN.GHG.CO2.PC.CE.AR5": "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
    "EN.POP.SLUM.UR.ZS": "Population living in slums (% of urban population)",
    "SL.TLF.CACT.ZS": "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)",
}

def get_all_countries(exclude_aggregates: bool = True) -> pd.DataFrame:
    """Download the World Bank 'countries' list. Optionally exclude aggregate regions."""
    url = "https://api.worldbank.org/v2/country"
    params = {"format": "json", "per_page": 400}
    rows = []
    session = requests.Session()

    # first call to get pages
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    meta, data = r.json()
    pages = meta.get("pages", 1)

    for p in range(1, pages + 1):
        params["page"] = p
        rr = session.get(url, params=params, timeout=60)
        rr.raise_for_status()
        _, dd = rr.json()
        for c in dd:
            region_val = (c.get("region") or {}).get("value")
            if exclude_aggregates and region_val == "Aggregates":
                continue
            rows.append({
                "Country Code": c["id"],
                "Country Name": c["name"],
            })
    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        raise RuntimeError("Failed to retrieve countries list from World Bank API.")
    return df

def fetch_indicator(ind_code: str, start: int, end: int, countries_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch a single indicator for ALL countries and years in [start, end]."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}"
    params = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}
    session = requests.Session()

    # first call to get pages
    r = session.get(url, params=params, timeout=90)
    r.raise_for_status()
    meta, data = r.json()
    if not isinstance(meta, dict):
        raise RuntimeError(f"Unexpected API response for {ind_code}")
    pages = meta.get("pages", 1)

    recs = []
    for page in range(1, pages + 1):
        params["page"] = page
        rr = session.get(url, params=params, timeout=90)
        rr.raise_for_status()
        _, dd = rr.json()
        if not dd:
            continue
        for row in dd:
            if not row:
                continue
            cc = row.get("countryiso3code")
            yr = row.get("date")
            try:
                yr = int(yr)
            except Exception:
                continue
            recs.append({
                "Country Code": cc,
                "Country Name": (row.get("country") or {}).get("value"),
                "Year": yr,
                ind_code: row.get("value"),
            })
    df = pd.DataFrame(recs)
    if df.empty:
        # Return an empty frame with expected columns to keep merge behavior simple
        return pd.DataFrame(columns=["Country Code", "Country Name", "Year", ind_code])

    # Keep only requested countries (excludes aggregates if countries_df excludes them)
    df = df[df["Country Code"].isin(countries_df["Country Code"])]
    # Deduplicate by country-year (keep last occurrence)
    df = df.sort_values(["Country Code", "Year"]).drop_duplicates(["Country Code", "Year"], keep="last")
    return df[["Country Code", "Country Name", "Year", ind_code]]

def build_dataset(start: int, end: int, include_aggregates: bool) -> pd.DataFrame:
    countries = get_all_countries(exclude_aggregates=not include_aggregates)

    # Use Population as the base grid (covers most country-years)
    base_code = "SP.POP.TOTL"
    if base_code not in INDICATORS:
        raise KeyError("SP.POP.TOTL missing from INDICATORS mapping.")
    base_df = fetch_indicator(base_code, start, end, countries)

    # Merge other indicators
    df = base_df.copy()
    for code in INDICATORS:
        if code == base_code:
            continue
        part = fetch_indicator(code, start, end, countries)
        df = df.merge(part[["Country Code", "Year", code]], on=["Country Code", "Year"], how="left")

    # Standardize country names using the countries reference
    df = df.merge(countries, on="Country Code", how="left", suffixes=("", "_official"))
    df["Country Name"] = df["Country Name_official"].fillna(df["Country Name"])
    df = df.drop(columns=["Country Name_official"])

    # Rename indicator columns to English labels
    rename_map = {code: INDICATORS[code] for code in INDICATORS}
    df = df.rename(columns=rename_map)

    # Reorder
    ordered_cols = ["Country Name", "Country Code", "Year"] + [rename_map[c] for c in INDICATORS.keys()]
    df = df[ordered_cols].sort_values(["Country Code", "Year"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Build a single-sheet Excel with selected World Bank indicators for all countries, 2000–2024.")
    ap.add_argument("--start", type=int, default=DEFAULT_START, help="Start year (default: 2000)")
    ap.add_argument("--end", type=int, default=DEFAULT_END, help="End year (default: 2024)")
    ap.add_argument("--out", type=str, default="worldbank_2000_2024.xlsx", help="Output Excel path")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    ap.add_argument("--include-aggregates", action="store_true", help="Include aggregates/regions (default: exclude)")

    args = ap.parse_args()

    if args.start > args.end:
        print("Error: --start must be <= --end", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Building dataset for years {args.start}–{args.end} (aggregates: {'INCLUDED' if args.include_aggregates else 'excluded'})")
    df = build_dataset(args.start, args.end, include_aggregates=args.include_aggregates)

    # Write Excel (single sheet)
    print(f"[INFO] Writing Excel -> {args.out}")
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="WDI_2000_2024")
        try:
            ws = writer.sheets["WDI_2000_2024"]
            ws.set_column(0, 0, 28)   # Country Name
            ws.set_column(1, 1, 12)   # Country Code
            ws.set_column(2, 2, 8)    # Year
            # rest
            ws.set_column(3, len(df.columns)-1, 36)
        except Exception:
            pass

    if args.csv:
        print(f"[INFO] Writing CSV -> {args.csv}")
        df.to_csv(args.csv, index=False)

    print(f"[DONE] Rows: {len(df):,} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
