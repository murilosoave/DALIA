# Copyright 2024-2025 DALIA authors. All rights reserved.

"""Build DALIA inputs for the GHCN-Daily global temperature example.

The model follows Lindgren et al. (2024), "A diffusion-based spatio-temporal
extension of Gaussian Matérn fields", Section 5.2: the daily mean of TMIN and
TMAX at the GHCN-Daily stations of one year, modelled with

    y = intercept + alpha * elevation + latitude-season basis + u(s, t) + e

where u(s, t) is DALIA's spatio-temporal SPDE field on the unit sphere with one
temporal knot per day, and e is Gaussian noise.

Raw data (downloaded separately, see README.MD):
    <raw-dir>/<year>.csv.gz        GHCN-Daily "by_year" file
    <raw-dir>/ghcnd-stations.txt   station list (id, lat, lon, elevation)

Outputs (layout of the other gst_* examples) in <out-dir>/inputs_<name>/:
    inputs_spatio_temporal/{c0,g1,g2,g3,m0,m1,m2,a}.npz
    inputs_spatial/{c0,g1,g2,a}.npz        (static field on the same mesh)
    inputs_regression/a.npz
    y.npy
    meta/                                  stations, days, mesh, meta.json

Example, small test case (a few hundred US stations, two months, coarse mesh):
    python prepare_inputs.py --raw-dir /path/to/raw --name small \
        --region conus --n-stations 300 --days 1 60 --mesh-nodes 150

Full case of the paper (all stations, whole year, ~1250 node mesh):
    python prepare_inputs.py --raw-dir /path/to/raw --name full --mesh-nodes 1250
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, cKDTree

EARTH_RADIUS_KM = 6371.0

REGIONS = {
    # name: (lat_min, lat_max, lon_min, lon_max)
    "global": (-90.0, 90.0, -180.0, 180.0),
    "conus": (24.0, 50.0, -125.0, -66.0),
    "europe": (35.0, 72.0, -11.0, 40.0),
    "australia": (-44.0, -10.0, 112.0, 154.0),
}


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def read_stations(path: Path) -> dict:
    """GHCN-Daily station list (fixed width): id -> (lat, lon, elevation_m)."""
    stations = {}
    with open(path) as fh:
        for line in fh:
            sid = line[0:11]
            lat = float(line[12:20])
            lon = float(line[21:30])
            elev = float(line[31:37])
            if elev <= -999.0:  # missing elevation
                continue
            stations[sid] = (lat, lon, elev)
    return stations


def read_daily_tmean(csv_gz: Path, cache: Path) -> tuple:
    """Daily mean of TMIN and TMAX (deg C) per station and day of year.

    Keeps only values without a quality flag and days where both TMIN and TMAX
    are present. The result is cached as an .npz next to the raw file.

    Returns (station_ids: array of str, doy: int array, tmean: float array).
    """
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["station"], z["doy"], z["tmean"]

    print(f"Parsing {csv_gz} (one pass over the whole year, a few minutes)...")
    t0 = time.time()
    tmax, tmin = {}, {}
    with gzip.open(csv_gz, "rt") as fh:
        for line in fh:
            f = line.split(",")
            el = f[2]
            if el != "TMAX" and el != "TMIN":
                continue
            if f[5] != "":  # quality flag set
                continue
            key = (f[0], f[1])
            (tmax if el == "TMAX" else tmin)[key] = int(f[3])
    keys = sorted(set(tmax) & set(tmin))
    station = np.array([k[0] for k in keys])
    dates = np.array([k[1] for k in keys])
    year = int(dates[0][:4])
    jan1 = np.datetime64(f"{year}-01-01")
    d = np.array([f"{s[:4]}-{s[4:6]}-{s[6:8]}" for s in dates], dtype="datetime64[D]")
    doy = (d - jan1).astype(int) + 1
    tmean = np.array([(tmax[k] + tmin[k]) / 20.0 for k in keys])  # tenths of deg C
    print(f"  {len(keys)} station-days with TMIN and TMAX, {time.time() - t0:.0f} s")
    np.savez(cache, station=station, doy=doy, tmean=tmean)
    return station, doy, tmean


# ---------------------------------------------------------------------------
# Sphere geometry
# ---------------------------------------------------------------------------
def lonlat_to_xyz(lon_deg, lat_deg):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def xyz_to_lonlat(xyz):
    lat = np.rad2deg(np.arcsin(np.clip(xyz[:, 2], -1, 1)))
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lon, lat


def fibonacci_sphere(n: int) -> np.ndarray:
    """n quasi-uniform points on the unit sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


def in_bbox(lon, lat, bbox, margin_deg=0.0):
    lat0, lat1, lon0, lon1 = bbox
    return (
        (lat >= lat0 - margin_deg)
        & (lat <= lat1 + margin_deg)
        & (lon >= lon0 - margin_deg)
        & (lon <= lon1 + margin_deg)
    )


def sphere_mesh(n_nodes: int, bbox, outer_nodes: int, margin_deg: float):
    """Triangulated unit sphere with ~n_nodes nodes inside bbox.

    For the global region the mesh is a Fibonacci lattice of n_nodes points.
    For a regional study the lattice is made dense enough to place n_nodes
    points inside the (enlarged) bbox and a coarse lattice of outer_nodes
    points covers the rest of the sphere, so that the triangulation (the
    convex hull of the points) is still a closed surface.
    """
    if bbox == REGIONS["global"]:
        loc = fibonacci_sphere(n_nodes)
    else:
        # fraction of the sphere area covered by the enlarged bbox
        lat0, lat1, lon0, lon1 = bbox
        lat0, lat1 = max(lat0 - margin_deg, -90), min(lat1 + margin_deg, 90)
        lon0, lon1 = max(lon0 - margin_deg, -180), min(lon1 + margin_deg, 180)
        frac = (
            (np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0)))
            * (lon1 - lon0)
            / 360.0
            / 2.0
        )
        fine = fibonacci_sphere(int(np.ceil(n_nodes / frac)))
        lon, lat = xyz_to_lonlat(fine)
        fine = fine[in_bbox(lon, lat, bbox, margin_deg)]
        coarse = fibonacci_sphere(outer_nodes)
        lon, lat = xyz_to_lonlat(coarse)
        # keep the coarse points away from the fine region
        coarse = coarse[~in_bbox(lon, lat, bbox, margin_deg + 5.0)]
        loc = np.vstack([fine, coarse])
    tv = ConvexHull(loc).simplices
    return loc, tv


def spherical_triangle_area(p):
    """Area of spherical triangles with unit-vector vertices p (n, 3, 3)."""
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    num = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    den = (
        1
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2 * np.arctan2(num, den)


def fem_matrices(loc, tv):
    """FEM matrices of fmesher on the sphere: c0 (lumped mass from the
    spherical triangle areas), g1 (stiffness of the flat triangles),
    g2 = g1 c0^-1 g1, g3 = g2 c0^-1 g1. Verified against fmesher::fm_fem."""
    ns = loc.shape[0]
    p = loc[tv]
    e = np.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]], axis=1)
    flat_area = 0.5 * np.linalg.norm(np.cross(e[:, 0], e[:, 1]), axis=1)
    local = np.einsum("tik,tjk->tij", e, e) / (4 * flat_area)[:, None, None]
    rows = np.repeat(tv, 3, axis=1).ravel()
    cols = np.tile(tv, (1, 3)).ravel()
    g1 = sp.csc_matrix((local.ravel(), (rows, cols)), shape=(ns, ns))
    g1.sum_duplicates()
    g1 = 0.5 * (g1 + g1.T)

    area = spherical_triangle_area(p)
    lumped = np.bincount(tv.ravel(), weights=np.repeat(area / 3, 3), minlength=ns)
    c0 = sp.diags(lumped, format="csc")
    c0_inv = sp.diags(1.0 / lumped, format="csc")

    g2 = (g1 @ c0_inv @ g1).tocsc()
    g2 = 0.5 * (g2 + g2.T)
    g3 = (g2 @ c0_inv @ g1).tocsc()
    g3 = 0.5 * (g3 + g3.T)
    return c0, g1, g2, g3


def spatial_basis(loc, tv, points):
    """Barycentric weights of `points` (unit vectors) in the triangles hit by the
    ray from the origin. Returns (triangle index, weights (n, 3))."""
    p = loc[tv]
    centroids = p.mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1)[:, None]
    tree = cKDTree(centroids)
    n = points.shape[0]
    tri = np.empty(n, dtype=int)
    w = np.empty((n, 3))
    for k in (8, 64, p.shape[0]):
        _, cand = tree.query(points, k=min(k, p.shape[0]))
        cand = np.atleast_2d(cand)
        todo = np.ones(n, dtype=bool) if k == 8 else ~done
        for i in np.flatnonzero(todo):
            for t in cand[i]:
                lam = np.linalg.solve(p[t].T, points[i])
                if (lam >= -1e-10).all():
                    tri[i], w[i] = t, lam / lam.sum()
                    todo[i] = False
                    break
        done = ~todo
        if done.all():
            break
    if not done.all():
        raise RuntimeError(f"{(~done).sum()} points not located in the mesh")
    return tri, w


def temporal_matrices(nt: int):
    """1D FEM matrices for knots 0, ..., nt-1 (unit spacing), as fmesher's
    fm_fem(fm_mesh_1d(1:nt)): m0 lumped mass (trapezoid weights), m1 half the
    boundary matrix, m2 stiffness (second differences)."""
    w = np.ones(nt)
    w[0] = w[-1] = 0.5
    m0 = sp.diags(w, format="csc")
    b = np.zeros(nt)
    b[0] = b[-1] = 0.5
    m1 = sp.diags(b, format="csc")
    main = 2 * np.ones(nt)
    main[0] = main[-1] = 1.0
    m2 = sp.diags([-np.ones(nt - 1), main, -np.ones(nt - 1)], [-1, 0, 1], format="csc")
    return m0, m1, m2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True, help="directory with <year>.csv.gz and ghcnd-stations.txt")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--name", required=True, help="inputs are written to <out-dir>/inputs_<name>")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--region", default="global", help=f"one of {list(REGIONS)} or 'lat0,lat1,lon0,lon1'")
    ap.add_argument("--n-stations", type=int, default=None, help="random subset of the stations (default: all)")
    ap.add_argument("--min-days", type=int, default=30, help="drop stations with fewer observed days in the window")
    ap.add_argument("--days", type=int, nargs=2, default=(1, 365), metavar=("FIRST", "LAST"), help="day-of-year window, inclusive")
    ap.add_argument("--mesh-nodes", type=int, default=1250, help="mesh nodes (inside the region for a regional run)")
    ap.add_argument("--outer-nodes", type=int, default=80, help="coarse nodes outside a regional mesh")
    ap.add_argument("--mesh-margin", type=float, default=5.0, help="degrees of fine mesh beyond the region bbox")
    ap.add_argument("--covariates", choices=["auto", "basic", "full"], default="auto",
                    help="basic: intercept, elevation, lat, lat^2; full: + seasonal x latitude basis (6 columns). auto: full if the window spans >= 120 days")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bbox = REGIONS[args.region] if args.region in REGIONS else tuple(float(v) for v in args.region.split(","))
    out = args.out_dir / f"inputs_{args.name}"
    rng = np.random.default_rng(args.seed)

    # --- observations
    stations = read_stations(args.raw_dir / "ghcnd-stations.txt")
    station, doy, tmean = read_daily_tmean(
        args.raw_dir / f"{args.year}.csv.gz", args.raw_dir / f"tmean_{args.year}.npz"
    )
    first, last = args.days
    keep = (doy >= first) & (doy <= last) & np.isin(station, list(stations))
    station, doy, tmean = station[keep], doy[keep], tmean[keep]

    ids, inv = np.unique(station, return_inverse=True)
    lat = np.array([stations[s][0] for s in ids])
    lon = np.array([stations[s][1] for s in ids])
    elev = np.array([stations[s][2] for s in ids])
    counts = np.bincount(inv, minlength=len(ids))
    sel = in_bbox(lon, lat, bbox) & (counts >= args.min_days)
    sel_idx = np.flatnonzero(sel)
    if args.n_stations is not None and args.n_stations < sel_idx.size:
        sel_idx = np.sort(rng.choice(sel_idx, args.n_stations, replace=False))
    keep = np.isin(inv, sel_idx)
    station, doy, tmean = station[keep], doy[keep], tmean[keep]
    ids, inv = np.unique(station, return_inverse=True)
    lat = np.array([stations[s][0] for s in ids])
    lon = np.array([stations[s][1] for s in ids])
    elev = np.array([stations[s][2] for s in ids])
    n_obs, n_sta = tmean.size, ids.size
    nt = last - first + 1
    print(f"{n_sta} stations, {nt} days, {n_obs} observations")

    # --- spatial mesh and FEM matrices
    loc, tv = sphere_mesh(args.mesh_nodes, bbox, args.outer_nodes, args.mesh_margin)
    ns = loc.shape[0]
    c0, g1, g2, g3 = fem_matrices(loc, tv)
    edge = np.linalg.norm(loc[tv[:, 0]] - loc[tv[:, 1]], axis=1)
    print(f"mesh: {ns} nodes, {tv.shape[0]} triangles, median edge {np.median(edge) * EARTH_RADIUS_KM:.0f} km")

    # --- temporal matrices
    m0, m1, m2 = temporal_matrices(nt)

    # --- projection matrices
    xyz = lonlat_to_xyz(lon, lat)
    tri, w = spatial_basis(loc, tv, xyz)
    s_cols = tv[tri]  # (n_sta, 3)
    t_idx = doy - first
    rows = np.repeat(np.arange(n_obs), 3)
    a_st = sp.csr_matrix(
        (w[inv].ravel(), (rows, (s_cols[inv] + (t_idx * ns)[:, None]).ravel())),
        shape=(n_obs, ns * nt),
    )
    a_sp = sp.csr_matrix((w[inv].ravel(), (rows, s_cols[inv].ravel())), shape=(n_obs, ns))

    # --- regression design: intercept, elevation (km), latitude-season basis
    full = args.covariates == "full" or (args.covariates == "auto" and nt >= 120)
    sl = np.sin(np.deg2rad(lat))[inv]
    cols = [np.ones(n_obs), elev[inv] / 1000.0, sl, sl**2]
    names = ["intercept", "elevation_km", "sin_lat", "sin_lat^2"]
    if full:
        ang = 2 * np.pi * (doy - 1) / 365.0
        cols += [sl * np.cos(ang), sl**2 * np.cos(ang), sl * np.sin(ang), sl**2 * np.sin(ang)]
        names += ["sin_lat*cos", "sin_lat^2*cos", "sin_lat*sin", "sin_lat^2*sin"]
    a_reg = sp.csr_matrix(np.column_stack(cols))

    # --- write
    for d in ["inputs_spatio_temporal", "inputs_spatial", "inputs_regression", "meta"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    st = out / "inputs_spatio_temporal"
    for nm, m in [("c0", c0), ("g1", g1), ("g2", g2), ("g3", g3), ("m0", m0), ("m1", m1), ("m2", m2), ("a", a_st)]:
        sp.save_npz(st / f"{nm}.npz", sp.csc_matrix(m))
    for nm, m in [("c0", c0), ("g1", g1), ("g2", g2), ("a", a_sp)]:
        sp.save_npz(out / "inputs_spatial" / f"{nm}.npz", sp.csc_matrix(m))
    sp.save_npz(out / "inputs_regression" / "a.npz", sp.csc_matrix(a_reg))
    np.save(out / "y.npy", tmean)

    meta = out / "meta"
    np.save(meta / "mesh_loc.npy", loc)
    np.save(meta / "mesh_tv.npy", tv)
    np.save(meta / "obs_station.npy", inv)
    np.save(meta / "obs_doy.npy", doy)
    with open(meta / "stations.csv", "w") as fh:
        fh.write("id,lat,lon,elevation_m\n")
        for s, la, lo, el in zip(ids, lat, lon, elev):
            fh.write(f"{s},{la},{lo},{el}\n")
    info = {
        "year": args.year,
        "region": args.region,
        "bbox": bbox,
        "days": [first, last],
        "n_stations": int(n_sta),
        "n_observations": int(n_obs),
        "ns": int(ns),
        "nt": int(nt),
        "n_latent_spatio_temporal": int(ns * nt),
        "n_fixed_effects": len(names),
        "fixed_effects": names,
        "median_edge_km": float(np.median(edge) * EARTH_RADIUS_KM),
        "y_units": "deg C, daily mean of TMIN and TMAX",
        "range_units": "radians on the unit sphere (1 rad = 6371 km); 600 km = 0.094",
        "seed": args.seed,
    }
    with open(meta / "meta.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"written to {out}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
