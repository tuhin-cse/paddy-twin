"""Experiment 2 (initial validation on real Sentinel-2): Boro rice 2023-24, Bangladesh.

Applies the simulation-trained background-aware retriever (scripts/01_...) to a
real Sentinel-2 L2A time series over a 10 km x 10 km Boro rice area in
Jamalpur/Sherpur (MGRS 45RYH), Nov 2023 - May 2024.

There is no field water-level data for this site, so the checks are internal
consistency and physical plausibility, not accuracy:
  V1 sim-to-real: do real rice spectra fall inside the simulated envelope?
  V2 anchors:     SCL open water -> ponded; pre-season dry fields -> not ponded.
  V3 season:      ponded fraction should be high at transplanting (Jan-Feb) and
                  fall as the crop matures and fields are drained (Apr-May).
  V4 LAI bias:    soil-only vs background-aware LAI on pixels flagged ponded
                  (the real-data counterpart of simulation E2).
  V5 AWD events:  wet -> dry -> wet transitions observed while LAI < 2, i.e.
                  inside the regime where simulation says optics can see water.

Usage:
    python scripts/02_real_s2_boro_jamalpur.py            # AWS mirror (default)
    python scripts/02_real_s2_boro_jamalpur.py --source pc  # Planetary Computer
"""

import argparse
import json
import os
import pathlib
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from sklearn.neighbors import NearestNeighbors

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import data_sources  # noqa: E402
from awdrt.s2 import FEATURE_BANDS, indices  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
# Remote COG reads: skip S3 directory listings and merge range requests.
for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
             "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "GDAL_HTTP_MULTIPLEX": "YES"}.items():
    os.environ.setdefault(k, v)
CACHE = ROOT / "data" / "cache"

TILE = "45RYH"
EPSG = 32645
# 10 km x 10 km, centre 89.985E 25.205N (UTM 45N), chosen as the most
# rice-dense window of the tile from a coarse transplant-flood + green-up screen.
CX, CY, HALF = 800780.0, 2791040.0, 5000.0
BOUNDS = (CX - HALF, CY - HALF, CX + HALF, CY + HALF)
BBOX_LL = [89.935, 25.160, 90.035, 25.250]
START, END = "2023-11-15", "2024-05-31"
MAX_TILE_CLOUD = 70
PX = 20.0
N = int(2 * HALF / PX)  # 500 x 500 pixels at 20 m
VALID_SCL = (4, 5, 6)  # vegetation, bare soil, water


def read_item(item):
    """Read one scene over BOUNDS on the 20 m grid; returns (bands, scl) or None."""
    cache = CACHE / f"{item['id']}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["R"], z["scl"]
    R = np.full((len(FEATURE_BANDS), N, N), np.nan, dtype=np.float32)
    with rasterio.open(item["hrefs"]["SCL"]) as s:
        scl = s.read(1, window=from_bounds(*BOUNDS, s.transform), out_shape=(N, N),
                     resampling=Resampling.nearest)
    if not np.isin(scl, VALID_SCL).any():
        return None
    for k, b in enumerate(FEATURE_BANDS):
        with rasterio.open(item["hrefs"][b]) as s:
            a = s.read(1, window=from_bounds(*BOUNDS, s.transform), out_shape=(N, N),
                       resampling=Resampling.average).astype(np.float32)
        R[k] = np.where(a > 0, a * item["scale"] + item["offset"], np.nan)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, R=R, scl=scl)
    return R, scl


def features(X):
    ix = indices(X)
    return np.column_stack([X, ix["ndvi"], ix["lswi"], ix["mndwi"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["aws", "pc"], default="aws")
    args = ap.parse_args()

    if args.source == "aws":
        items = data_sources.aws_items(TILE, START, END)
    else:
        items = data_sources.pc_items(BBOX_LL, START, END, mgrs_tile=TILE)
    items = [i for i in items if i["cloud"] is None or i["cloud"] < MAX_TILE_CLOUD]
    print(f"{len(items)} scenes with tile cloud < {MAX_TILE_CLOUD}% ({args.source})")

    with open(OUT / "models" / "background_aware_models.pkl", "rb") as f:
        M = pickle.load(f)

    # ── read the stack, one observation per date (first valid pixel wins) ────
    dates, stack, scls = [], [], []
    for it in items:
        got = read_item(it)
        if got is None:
            continue
        R, scl = got
        if dates and dates[-1] == it["date"]:
            fill = ~np.isin(scls[-1], VALID_SCL) & np.isin(scl, VALID_SCL)
            stack[-1][:, fill] = R[:, fill]
            scls[-1][fill] = scl[fill]
            continue
        dates.append(it["date"])
        stack.append(R)
        scls.append(scl)
        print(" read", it["id"], f"valid={np.isin(scl, VALID_SCL).mean():.2f}")
    T = len(dates)
    stack = np.stack(stack)            # (T, bands, N, N)
    scls = np.stack(scls)              # (T, N, N)
    valid = np.isin(scls, VALID_SCL) & np.isfinite(stack).all(axis=1)
    dts = pd.to_datetime(dates)

    # ── apply the retriever to every valid pixel-date ───────────────────────
    X = np.moveaxis(stack, 1, -1)[valid]              # (n, bands)
    F = features(X)
    p_pond = np.full(valid.shape, np.nan, np.float32)
    lai = np.full(valid.shape, np.nan, np.float32)
    lai_soil = np.full(valid.shape, np.nan, np.float32)
    cls3 = np.full(valid.shape, "", dtype=object)
    p_pond[valid] = M["clf_ponded"].predict_proba(F)[:, 1]
    lai[valid] = M["reg_lai"].predict(F)
    lai_soil[valid] = M["reg_lai_soil_only"].predict(F)
    cls3[valid] = M["clf_3class"].predict(F)
    ix = indices(np.moveaxis(stack, 1, -1))
    ndvi, evi, lswi = ix["ndvi"], ix["evi"], ix["lswi"]

    # ── rice mask: transplant flood (Xiao) in Dec-Feb, then NDVI > 0.6 by Apr ─
    early = (dts >= "2023-12-15") & (dts <= "2024-02-29")
    late = (dts >= "2024-03-01") & (dts <= "2024-04-30")
    flood_sig = ((lswi + 0.05 >= evi) & (ndvi > 0.05) & (ndvi < 0.4) & valid)[early].any(axis=0)
    green = np.nanmax(np.where(valid[late], ndvi[late], np.nan), axis=0) > 0.6
    open_water_always = (scls == 6).sum(0) >= 0.8 * valid.sum(0)
    rice = flood_sig & green & ~open_water_always
    print(f"rice pixels: {rice.sum()} ({rice.mean():.1%} of window)")

    res = {"source": args.source, "tile": TILE, "bounds_utm45n": BOUNDS, "dates": dates,
           "n_dates": T, "rice_fraction": float(rice.mean())}

    # V1 sim-to-real envelope: nearest-neighbour distance of real rice spectra to
    # the simulated set, compared with the sim-to-sim NN distance distribution.
    sim = pd.read_csv(OUT / "sim_test_spectra.csv.gz")
    S = sim[FEATURE_BANDS].to_numpy()
    nn = NearestNeighbors(n_neighbors=2).fit(S)
    d_ss = nn.kneighbors(S[:5000])[0][:, 1]
    thr = float(np.quantile(d_ss, 0.99))
    rice_obs = np.moveaxis(stack, 1, -1)[valid & rice[None]]
    sub = rice_obs[np.random.default_rng(0).choice(len(rice_obs), min(20000, len(rice_obs)), replace=False)]
    d_rs = nn.kneighbors(sub, n_neighbors=1)[0][:, 0]
    res["V1_sim_to_real"] = {"sim_nn_dist_q99": thr, "real_rice_nn_dist_median": float(np.median(d_rs)),
                             "real_rice_within_sim_q99": float((d_rs <= thr).mean())}

    # V2 anchors.
    ow = valid & (scls == 6) & (ndvi < 0)
    pre = (dts < "2023-12-10")
    dry_bare = valid & (scls == 5) & pre[:, None, None]
    res["V2_anchors"] = {
        "open_water_n": int(ow.sum()),
        "open_water_mean_p_ponded": float(np.nanmean(p_pond[ow])),
        "preseason_bare_n": int(dry_bare.sum()),
        "preseason_bare_mean_p_ponded": float(np.nanmean(p_pond[dry_bare])),
    }

    # V3 season curves over rice pixels.
    rows = []
    for t in range(T):
        m = valid[t] & rice
        if m.sum() < 0.3 * rice.sum():
            continue  # too cloudy over the rice to summarise
        rows.append({"date": dates[t], "valid_rice_frac": float(m.sum() / rice.sum()),
                     "ponded_frac": float((p_pond[t][m] > 0.5).mean()),
                     "ponded_frac_lai_lt2": float((p_pond[t][m & (lai[t] < 2)] > 0.5).mean())
                     if (m & (lai[t] < 2)).any() else np.nan,
                     "xiao_flood_frac": float((lswi[t] + 0.05 >= evi[t])[m].mean()),
                     "lai_median": float(np.nanmedian(lai[t][m])),
                     "lai_p25": float(np.nanpercentile(lai[t][m], 25)),
                     "lai_p75": float(np.nanpercentile(lai[t][m], 75)),
                     "ndvi_median": float(np.nanmedian(ndvi[t][m])),
                     # Background class only where simulation says it is visible
                     # (LAI < 2); above that the classifier has no signal and
                     # would otherwise fall back to its prior.
                     "undetermined_frac": float((lai[t][m] >= 2).mean()),
                     "saturated_frac": float(((cls3[t] == "saturated") & (lai[t] < 2))[m].mean()),
                     "drying_frac": float(((cls3[t] == "drying") & (lai[t] < 2))[m].mean())})
    season = pd.DataFrame(rows)
    season.to_csv(OUT / "real_season_curve.csv", index=False)
    res["V3_season"] = rows

    # V4 LAI bias on pixels the retriever calls ponded with LAI < 2.
    pond_early = valid & rice[None] & (p_pond > 0.8) & (lai < 2)
    res["V4_lai_ponded"] = {
        "n": int(pond_early.sum()),
        "mean_lai_background_aware": float(np.nanmean(lai[pond_early])),
        "mean_lai_soil_only": float(np.nanmean(lai_soil[pond_early])),
        "mean_difference_soil_minus_aware": float(np.nanmean(lai_soil[pond_early] - lai[pond_early])),
    }

    # V5 candidate AWD dry-downs: per rice pixel, clear dates only, while LAI < 2:
    # ponded (p > 0.7) -> not ponded (p < 0.3) -> ponded again, within 30 days.
    growing = (dts >= "2024-01-01") & (dts <= "2024-04-15")
    obs_ok = valid & (lai < 2) & growing[:, None, None]
    state = np.where(obs_ok & (p_pond > 0.7), 1, np.where(obs_ok & (p_pond < 0.3), 0, -1))
    rr, cc = np.nonzero(rice)
    n_events = np.zeros(rr.size, int)
    dur = []
    day = (dts - dts[0]).days.to_numpy()
    for k, (r, c) in enumerate(zip(rr, cc)):
        seq = [(day[t], s) for t, s in enumerate(state[:, r, c]) if s >= 0]
        i = 0
        while i < len(seq) - 2:
            if seq[i][1] == 1 and seq[i + 1][1] == 0:
                j = i + 1
                while j < len(seq) and seq[j][1] == 0:
                    j += 1
                if j < len(seq) and seq[j][0] - seq[i][0] <= 30:
                    n_events[k] += 1
                    dur.append(seq[j][0] - seq[i][0])
                i = j
            else:
                i += 1
    obs_per_px = obs_ok[:, rr, cc].sum(0)
    res["V5_awd_candidates"] = {
        "rice_pixels": int(rr.size),
        "median_clear_obs_with_lai_lt2": float(np.median(obs_per_px)),
        "share_pixels_with_>=1_event": float((n_events >= 1).mean()),
        "median_wet_to_wet_days": float(np.median(dur)) if dur else None,
    }
    ev_map = np.full(rice.shape, np.nan)
    ev_map[rr, cc] = n_events
    # Spatial coherence: AWD is managed per field, so real dry-downs should come
    # in patches; classifier noise would be salt-and-pepper. Compare how often an
    # event pixel's 8 neighbours (rice only) also have events against the base rate.
    ev = np.nan_to_num(ev_map) >= 1
    nb_ev = sum(np.roll(np.roll(ev, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    nb_rice = sum(np.roll(np.roll(rice, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    inner = rice & (nb_rice == 8)
    res["V5_awd_candidates"]["neighbour_event_rate_given_event"] = float((nb_ev[inner & ev] / 8).mean())
    res["V5_awd_candidates"]["base_event_rate"] = float(ev[rice].mean())

    (OUT / "real_validation_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("V3_season", "dates")}, indent=2))
    print(season.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    plot(season, dts, stack, scls, valid, p_pond, lai, rice, ev_map, dates)


def rgb(R):
    b = [FEATURE_BANDS.index(x) for x in ("B04", "B03", "B02")]
    img = np.moveaxis(R[b], 0, -1)
    return np.clip(np.nan_to_num(img) / 0.15, 0, 1)


def plot(season, dts, stack, scls, valid, p_pond, lai, rice, ev_map, dates):
    s = season.copy()
    s["date"] = pd.to_datetime(s["date"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.date, s.ponded_frac, "o-", label="standing water, RT retriever")
    ax.plot(s.date, s.xiao_flood_frac, "s--", label="flood, Xiao 2005 rule")
    ax.set_ylabel("share of rice pixels")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.fill_between(s.date, s.lai_p25, s.lai_p75, color="g", alpha=0.15)
    ax2.plot(s.date, s.lai_median, "g-", label="LAI (median, IQR)")
    ax2.set_ylabel("LAI", color="g")
    ax2.axhline(2, color="g", ls=":", lw=0.8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Boro 2023-24, Jamalpur/Sherpur (S2 L2A, 10x10 km)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig4_real_season_curve.png", dpi=150)
    plt.close(fig)

    # Map panel: a transplanting date, a tillering date, a late date + event map.
    want = ["2024-01-26", "2024-02-18", "2024-03-16"]
    picks = [dates.index(d) for d in want if d in dates]
    fig, axes = plt.subplots(2, len(picks) + 1, figsize=(4 * (len(picks) + 1), 8))
    for j, t in enumerate(picks):
        axes[0, j].imshow(rgb(stack[t]))
        axes[0, j].set_title(f"{dates[t]} true colour")
        pm = np.where(valid[t] & rice, p_pond[t], np.nan)
        im = axes[1, j].imshow(pm, cmap="Blues", vmin=0, vmax=1)
        axes[1, j].set_title("P(standing water), rice pixels")
    axes[0, -1].imshow(rice, cmap="Greens")
    axes[0, -1].set_title("rice mask")
    im2 = axes[1, -1].imshow(ev_map, cmap="magma_r", vmin=0, vmax=3)
    axes[1, -1].set_title("candidate AWD dry-downs (count)")
    fig.colorbar(im, ax=axes[1, :-1], shrink=0.6)
    fig.colorbar(im2, ax=axes[1, -1], shrink=0.6)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.savefig(FIG / "fig5_real_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
