# Copyright 2024-2025 DALIA authors. All rights reserved.

"""Collect the summary.json files of the sweep runs into one CSV.

    python collect_sweep.py [--out stiles_fullsolve.csv] [--inputs full]

The CSV (columns: case, inputs, ranks, threads, iterations, n, n_observations,
wall_time_s, hostname, slurm_job_id, outputs_dir) is what the poster figure
script reads (KAUST_POSTER/figs_dalia/stiles_fullsolve.csv). Several runs of
the same configuration are all kept; the figure script takes them as they come,
so remove the ones you do not want before copying the file.
"""

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["case", "inputs", "solver", "ranks", "threads", "iterations", "n", "n_observations",
           "wall_time_s", "hostname", "slurm_job_id", "outputs_dir"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "stiles_fullsolve.csv"))
    ap.add_argument("--inputs", default=None, help="keep only runs of this inputs name")
    ap.add_argument("--solver", default="stiles")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "outputs_*", "summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        if s.get("solver") != args.solver:
            continue
        if args.inputs and s.get("inputs") != args.inputs:
            continue
        s["outputs_dir"] = os.path.basename(os.path.dirname(path))
        rows.append({k: s.get(k, "") for k in COLUMNS})
    rows.sort(key=lambda r: (r["inputs"], int(r["ranks"] or 0), int(r["threads"] or 0)))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out}")
    for r in rows:
        print(f"  {r['inputs']:8s} ranks={r['ranks']:>3} threads={r['threads']:>4} "
              f"iterations={r['iterations']:>3} wall={float(r['wall_time_s']):8.1f} s")


if __name__ == "__main__":
    main()
