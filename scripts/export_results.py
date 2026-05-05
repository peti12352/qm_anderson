"""
Export λ_M summary tables from HDF5 files to CSV.

Run after any completed TMM run to produce git-trackable summaries:

    python scripts/export_results.py

Writes one CSV per dataset to results/:
    results/sc_benchmark.csv
    results/sc_dilution.csv
    results/bcc_dilution.csv

CSV columns: M, x_value, lambda_mean, lambda_stderr

These CSVs are committed to git. The raw HDF5 files (data/*.h5) are NOT -
they can be 100–400 MB each (raw Lyapunov exponent samples for bootstrap
reanalysis). Keep them on the compute machine or transfer separately.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config  # noqa
from src.io_utils import load_lambda

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("sc_benchmark.h5",  "sc_benchmark.csv"),
    ("sc_dilution.h5",   "sc_dilution.csv"),
    ("bcc_dilution.h5",  "bcc_dilution.csv"),
]


def export_dataset(h5_path: Path, csv_path: Path) -> None:
    data = load_lambda(h5_path)
    M_list   = [int(m) for m in data["M_list"]]
    x_values = data["x_values"]

    rows = []
    for M in M_list:
        for x in x_values:
            mean, err = data["lambda"][M][x]
            rows.append({
                "M":             M,
                "x_value":       float(x),
                "lambda_mean":   mean,
                "lambda_stderr": err,
            })

    # Write metadata as comment lines before the CSV header
    meta = data["metadata"]
    with open(csv_path, "w", newline="") as f:
        for k, v in meta.items():
            if k not in ("x_values", "M_list"):
                f.write(f"# {k}: {v}\n")
        writer = csv.DictWriter(f, fieldnames=["M", "x_value", "lambda_mean", "lambda_stderr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows):4d} rows → {csv_path}  "
          f"(M={M_list}, {len(x_values)} x-values)")


def main() -> None:
    found = 0
    for h5_name, csv_name in DATASETS:
        h5_path  = DATA_DIR / h5_name
        csv_path = RESULTS_DIR / csv_name
        if not h5_path.exists():
            print(f"  Skipping {h5_name} (not found)")
            continue
        found += 1
        export_dataset(h5_path, csv_path)

    if found == 0:
        print("No data files found in data/. Run benchmark/dilution scripts first.")
    else:
        print(f"\n{found} dataset(s) exported to {RESULTS_DIR}/")
        print("Now commit with:")
        print("  git add results/ && git commit -m 'Add run results' && git push")


if __name__ == "__main__":
    main()
