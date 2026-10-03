"""Experiment 4: can satellites learn each paddy's own water-loss rate (S&P)?

Part A, synthetic twin. 2,000 virtual Boro fields with known seepage+percolation
are irrigated by simulated farmers under real ERA5 weather. They are "observed"
on the real Sentinel-2 calendar and cloud pattern of the Jamalpur window, with
the wet/dry error rates measured in Experiment 1. S&P is then recovered under:
    S0 satellite only
    S1 + irrigation date logged by the farmer in the app
    S2 + irrigation date and pump hours (irrigation depth) logged
    S3 satellite only, 2-day revisit (S2A/B/C + Landsat/HLS + S1)
and compared with what an ET-only advisory (IRAS-style) or a regional constant
S&P would predict for the next irrigation.

Part B, real data. The same estimator runs (satellite only, S0) on 60 m blocks
of the Jamalpur window, and the S&P map is checked against an independent
pre-season soil proxy from bare-soil Sentinel-2 (Nov-Dec).

Requires results from scripts 01, 02 and 04a.
"""

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
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from awdrt import hydro  # noqa: E402
from awdrt.s2 import FEATURE_BANDS  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
T0 = pd.Timestamp("2024-01-15")
T1 = pd.Timestamp("2024-03-20")
LEAKY = 8.0  # mm/d, "leaky" field threshold for advice

wx = pd.read_csv(OUT / "era5_daily_jamalpur.csv", index_col=0, parse_dates=True)
DAY0 = wx.index[0]
ET0 = wx.et0.to_numpy()
RAIN = wx.rain.to_numpy()


def dayof(ts):
    return (pd.Timestamp(ts) - DAY0).days


def et0_mean(w0, w1):
    i0, i1 = int(np.floor(w0)), int(np.ceil(w1))
    return float(ET0[i0:max(i1, i0 + 1)].mean())


def rain_sum(w0, w1):
    i0, i1 = int(np.floor(w0)), int(np.ceil(w1))
    return float(RAIN[i0:i1].sum())


# ── Part A: synthetic twin ────────────────────────────────────────────────────

def detection_rates():
    e1 = json.loads((OUT / "simulation_metrics.json").read_text())["E1_background"]
    lo = np.array(e1["lai_bin_lo"])
    rec = np.array(e1["rt_recall_ponded"])
    spec = 2 * np.array(e1["rt_binary_bacc"]) - rec
    return lo, rec, spec


SEASON_END = pd.Timestamp("2024-04-15")
# S1 wet/dry accuracy once LAI >= 2. 0.85 was assumed; script 03b measured about 0.60 at LAI 2-3
# (results/s1_wetdry_accuracy.json). Override with SAR_ACC=0.60; outputs then get a "_sar060" suffix.
SAR_ACC = float(os.environ.get("SAR_ACC", 0.85))
RUN_TAG = "" if SAR_ACC == 0.85 else f"_sar{round(SAR_ACC * 100):03d}"

# Scenario: (seasons, SAR after canopy closure, app logs irrigation date,
#            app logs pump hours -> depth, app "water gone" reports)
SCENARIOS = {
    "S0 optical, 1 season":                  (1, False, False, False, False),
    "S1 + app irrigation date":              (1, False, True, False, False),
    "S2 + app date & pump hours":            (1, False, True, True, False),
    "S3 + app date, pump hours, water-gone": (1, False, True, True, True),
    "S4 optical+SAR, 1 season":              (1, True, False, False, False),
    "S5 optical+SAR, 3 seasons":             (3, True, False, False, False),
    "S6 optical+SAR+app (date, hours), 3 s": (3, True, True, True, False),
}


def lai_curve(t, transplant):
    return np.minimum(0.5 + np.clip(t - transplant, 0, None) / 40 * 1.5, 5.0)


def simulate_season(rng, sp, obs_days, obs_valid, rates, sar, topup_frac=0.0):
    """One Boro season of one field; returns observations and the app log.

    sp: S&P in mm/d, or a callable sp(h_mm, days_since_transplant) (Experiment 8).

    topup_frac: share of farmers who re-pump while water still stands (at
    5-20 mm, within a day), so their field only dries if it leaks faster than
    they react. Experiment 4c; the default reproduces Experiments 4 and 4b.
    """
    lo, rec, spec = rates
    transplant = dayof(T0) + rng.uniform(0, 20)
    awd = rng.random() < 0.3
    topup = topup_frac > 0 and rng.random() < topup_frac
    thr = rng.uniform(5, 20)
    dt_ = 0.05
    t = np.arange(transplant, dayof(SEASON_END), dt_)
    lai = lai_curve(t, transplant)
    h, wet = 0.0, np.zeros(t.size, bool)
    irr_t, irr_h, gone_t = [], [], []
    next_irr = transplant  # puddling flood at transplanting
    for k, tk in enumerate(t):
        day = int(tk)
        if next_irr is not None and tk >= next_irr:
            h0 = rng.lognormal(np.log(hydro.H0_MEDIAN), hydro.H0_SIGMA)
            h += h0
            irr_t.append(tk)
            irr_h.append(h0)
            next_irr = None
        sp_k = sp(h, tk - transplant) if callable(sp) else sp  # callable: Experiment 8's richer physics
        h += (RAIN[day] - hydro.kc_rice(lai[k]) * ET0[day] - sp_k) * dt_
        if topup and 0 < h < thr and next_irr is None:
            next_irr = tk + rng.uniform(0, 1)
        if h <= 0:
            if k and wet[k - 1]:
                gone_t.append(tk)
            h = 0.0
            if next_irr is None:  # field just ran dry: farmer waits, then pumps again
                next_irr = tk + (rng.uniform(3, 8) if awd else rng.uniform(0.5, 3))
        wet[k] = h > 0

    sel = (obs_days >= transplant) & (obs_days < dayof(SEASON_END))
    days, valid = obs_days[sel], rng.random(sel.sum()) < obs_valid[sel]
    idx = np.clip(np.searchsorted(t, days), 0, t.size - 1)
    lai_o, truth = lai[idx], wet[idx]
    b = np.clip(np.searchsorted(lo, lai_o, side="right") - 1, 0, len(lo) - 1)
    obs = np.where(truth, rng.random(days.size) < rec[b], rng.random(days.size) > spec[b])
    keep = valid & (lai_o < 2.0)
    days, obs = days[keep], obs[keep]
    if sar:  # Sentinel-1 every 6 days (S1A + S1C) once optics is blind
        closed = t[lai >= 2.0]
        if closed.size:
            sd = np.arange(closed[0] + rng.uniform(0, 6), t[-1], 6.0)
            si = np.searchsorted(t, sd)
            so = np.where(wet[si], rng.random(sd.size) < SAR_ACC, rng.random(sd.size) > SAR_ACC)
            days, obs = np.concatenate([days, sd]), np.concatenate([obs, so])
    o = np.argsort(days)
    return dict(days=days[o], state=obs[o].astype(int), irr_t=np.array(irr_t), irr_h=np.array(irr_h),
                gone_t=np.array(gone_t), transplant=transplant)


def fit_field(seasons, use_date, use_hours, use_gone, rng):
    eps_all, h0_all = [], []
    for s in seasons:
        days, state = s["days"], s["state"]
        if use_gone and s["gone_t"].size:  # farmer reports "no water" on some mornings
            rep = s["gone_t"][rng.random(s["gone_t"].size) < 0.5] + rng.uniform(0, 1, 1)
            days = np.concatenate([days, rep])
            state = np.concatenate([state, np.zeros(rep.size, int)])
        if use_date:  # the app log is also an observation: wet right after pumping
            days = np.concatenate([days, s["irr_t"] + 0.01])
            state = np.concatenate([state, np.ones(s["irr_t"].size, int)])
        o = np.argsort(days)
        eps = hydro.episodes(days[o], state[o], tau=s["irr_t"] if use_date else None)
        for e in eps:
            e["tp"] = s["transplant"]
        eps_all += eps
        if use_hours:
            h0_all += [s["irr_h"][np.argmin(abs(s["irr_t"] - e["w0"]))] for e in eps]
    if not eps_all:
        return None, 0

    def kc(w0, w1, _eps=eps_all):
        tp = next(e["tp"] for e in _eps if e["w0"] == w0)
        return float(hydro.kc_rice(lai_curve(np.array([(w0 + w1) / 2]), tp))[0])

    post = hydro.posterior(eps_all, et0_mean, rain_sum, kc=kc, h0=h0_all if use_hours else None)
    return hydro.summarize(post), len(eps_all)


def part_a(rng, n_fields=1000):
    real = json.loads((OUT / "real_validation_metrics.json").read_text())
    season = pd.read_csv(OUT / "real_season_curve.csv", index_col="date")
    obs_days = np.array([dayof(d) for d in real["dates"]], float)
    obs_valid = np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])
    rates = detection_rates()
    et_mid = float(wx.loc["2024-02-01":"2024-04-15"].et0.mean())

    rows = []
    for n in range(n_fields):
        fr = np.random.default_rng(10_000 + n)
        sp = float(np.clip(fr.lognormal(np.log(5.0), 0.6), 0.5, 30))
        # Year-to-year puddling quality moves S&P by ~10 %.
        cache = {}
        for name, (ns, sar, d_, h_, g_) in SCENARIOS.items():
            key = (ns, sar)
            if key not in cache:
                sr = np.random.default_rng(20_000 + n)
                cache[key] = [simulate_season(sr, sp * (1 + 0.1 * sr.standard_normal()), obs_days, obs_valid, rates, sar)
                              for _ in range(ns)]
            est, k = fit_field(cache[key], d_, h_, g_, np.random.default_rng(30_000 + n))
            rows.append({"field": n, "scenario": name, "sp_true": sp, "n_episodes": k,
                         "sp_hat": est["median"] if est else np.nan,
                         "q05": est["q05"] if est else np.nan, "q95": est["q95"] if est else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "twin_fields.csv.gz", index=False)

    # Decision quality: how many days until the next 50 mm irrigation has gone.
    kc = hydro.kc_rice(2.0)
    D_true = 50 / (kc * et_mid + df.sp_true)
    df["err_field"] = abs(50 / (kc * et_mid + df.sp_hat) - D_true)
    df["err_et_only"] = abs(50 / (kc * et_mid) - D_true)
    df["err_regional"] = abs(50 / (kc * et_mid + 5.0) - D_true)
    # Water: pumping on the ET-only or regional schedule vs. on the field's own
    # schedule. A late re-irrigation stresses the crop; an early one is wasted
    # water. Count the share of fields where the error exceeds 1 day either way.
    summ = {}
    for scen, g in df.groupby("scenario", sort=False):
        ok = g.dropna(subset=["sp_hat"])
        leaky_t, leaky_h = ok.sp_true > LEAKY, ok.sp_hat > LEAKY
        summ[scen] = {
            "fields_with_estimate": float(len(ok) / len(g)),
            "median_episodes": float(ok.n_episodes.median()),
            "median_abs_pct_error": float((abs(ok.sp_hat - ok.sp_true) / ok.sp_true).median() * 100),
            "spearman": float(spearmanr(ok.sp_true, ok.sp_hat).correlation),
            "ci90_coverage": float(((ok.q05 <= ok.sp_true) & (ok.sp_true <= ok.q95)).mean()),
            "leaky_detection_bacc": float(0.5 * ((leaky_t & leaky_h).sum() / max(leaky_t.sum(), 1)
                                                 + (~leaky_t & ~leaky_h).sum() / max((~leaky_t).sum(), 1))),
            "next_dry_err_days_field": float(ok.err_field.median()),
            "next_dry_err_days_et_only": float(ok.err_et_only.median()),
            "next_dry_err_days_regional_const": float(ok.err_regional.median()),
            "share_err_gt1d_field": float((ok.err_field > 1).mean()),
            "share_err_gt1d_regional": float((ok.err_regional > 1).mean()),
            "share_err_gt1d_et_only": float((ok.err_et_only > 1).mean()),
        }
    summ["et0_feb_apr15_mean"] = et_mid
    return df, summ


# ── Part B: real Jamalpur window ─────────────────────────────────────────────

def part_b():
    real = __import__("02_real_s2_boro_jamalpur")
    with open(OUT / "models" / "background_aware_models.pkl", "rb") as f:
        M = pickle.load(f)
    items = real.data_sources.aws_items(real.TILE, "2023-11-15", "2024-03-25")
    items = [i for i in items if i["cloud"] is None or i["cloud"] < real.MAX_TILE_CLOUD]
    by_date = {}
    for it in items:
        got = real.read_item(it)
        if got is None:
            continue
        R, scl = got
        ok = np.isin(scl, real.VALID_SCL) & np.isfinite(R).all(0)
        if it["date"] in by_date:
            prev = by_date[it["date"]]
            fill = ~prev[2] & ok
            prev[0][:, fill] = R[:, fill]
            prev[2][fill] = True
            continue
        by_date[it["date"]] = [R, scl, ok]

    dates = sorted(by_date)
    B = 3  # 3x3 blocks of 20 m = 60 m
    nb = real.N // B
    P, LAI, V = [], [], []
    for d in dates:
        R, scl, ok = by_date[d]
        X = np.moveaxis(R, 0, -1)
        p = np.full(ok.shape, np.nan, np.float32)
        lai = np.full(ok.shape, np.nan, np.float32)
        F = real.features(X[ok])
        p[ok] = M["clf_ponded"].predict_proba(F)[:, 1]
        lai[ok] = M["reg_lai"].predict(F)
        crop = lambda a: a[:nb * B, :nb * B].reshape(nb, B, nb, B)  # noqa: E731
        with np.errstate(all="ignore"):
            P.append(np.nanmean(crop(p), axis=(1, 3)))
            LAI.append(np.nanmean(crop(lai), axis=(1, 3)))
            V.append(crop(ok).mean(axis=(1, 3)))
    P, LAI, V = np.stack(P), np.stack(LAI), np.stack(V)
    dts = pd.to_datetime(dates)
    days = np.array([dayof(d) for d in dates], float)

    # Rice blocks: same rule as script 02, at block level.
    rice_px = np.zeros((real.N, real.N), bool)
    early = [d for d in dates if "2023-12-15" <= d <= "2024-02-29"]
    for d in early:
        R, scl, ok = by_date[d]
        from awdrt.s2 import indices
        ix = indices(np.moveaxis(R, 0, -1))
        rice_px |= (ix["lswi"] + 0.05 >= ix["evi"]) & (ix["ndvi"] > 0.05) & (ix["ndvi"] < 0.4) & ok
    rice_blk = rice_px[:nb * B, :nb * B].reshape(nb, B, nb, B).mean(axis=(1, 3)) > 0.7

    season = (dts >= T0) & (dts <= T1)
    sp_map = np.full((nb, nb), np.nan)
    width = np.full((nb, nb), np.nan)
    neps = np.zeros((nb, nb), int)
    for i, j in zip(*np.nonzero(rice_blk)):
        m = season & (V[:, i, j] > 0.8) & (LAI[:, i, j] < 2.0) & ((P[:, i, j] > 0.7) | (P[:, i, j] < 0.3))
        if m.sum() < 5:
            continue
        st = (P[m, i, j] > 0.7).astype(int)
        eps = hydro.episodes(days[m], st)
        if len(eps) < 2:
            continue
        s = hydro.summarize(hydro.posterior(eps, et0_mean, rain_sum, kc=hydro.kc_rice(1.0)))
        sp_map[i, j] = s["median"]
        width[i, j] = s["q95"] - s["q05"]
        neps[i, j] = len(eps)

    # Independent soil proxy: pre-season bare-soil composite (Nov 15 - Dec 31).
    pre = [d for d in dates if d <= "2023-12-31"]
    stack = []
    for d in pre:
        R, scl, ok = by_date[d]
        bare = ok & (scl == 5)
        stack.append(np.where(bare[None], R, np.nan))
    with np.errstate(all="ignore"):
        comp = np.nanmedian(np.stack(stack), axis=0)
    b11, b12 = comp[FEATURE_BANDS.index("B11")], comp[FEATURE_BANDS.index("B12")]
    blk = lambda a: np.nanmean(a[:nb * B, :nb * B].reshape(nb, B, nb, B), axis=(1, 3))  # noqa: E731
    with np.errstate(all="ignore"):
        swir = blk((b11 + b12) / 2)
        clay = blk(b11 / b12)
    ok = np.isfinite(sp_map) & np.isfinite(swir) & np.isfinite(clay)

    # Spatial coherence: correlation of each block's S&P with its 4-neighbour mean.
    nb_mean = np.nanmean(np.stack([np.roll(sp_map, s, a) for s in (-1, 1) for a in (0, 1)]), axis=0)
    coh = np.isfinite(sp_map) & np.isfinite(nb_mean)

    res = {
        "blocks_rice": int(rice_blk.sum()),
        "blocks_with_estimate": int(np.isfinite(sp_map).sum()),
        "median_episodes": float(np.median(neps[neps > 0])) if (neps > 0).any() else 0,
        "sp_median_mm_d": float(np.nanmedian(sp_map)),
        "sp_p10_p90": [float(np.nanpercentile(sp_map, 10)), float(np.nanpercentile(sp_map, 90))],
        "median_ci90_width": float(np.nanmedian(width)),
        "share_leaky_gt8": float((sp_map[np.isfinite(sp_map)] > LEAKY).mean()),
        "neighbour_spearman": float(spearmanr(sp_map[coh], nb_mean[coh]).correlation),
        "soil_proxy_n": int(ok.sum()),
        "spearman_sp_vs_bare_swir": float(spearmanr(sp_map[ok], swir[ok]).correlation),
        "spearman_sp_vs_clay_ratio_b11_b12": float(spearmanr(sp_map[ok], clay[ok]).correlation),
    }
    np.save(OUT / "real_sp_map_60m.npy", sp_map)
    return res, sp_map, swir


def plot(df, summ, sp_map, swir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    scen = list(SCENARIOS)
    for sc, c in zip((scen[0], scen[-1]), ("#999", "#1f77b4")):
        g = df[df.scenario == sc].dropna(subset=["sp_hat"])
        axes[0].scatter(g.sp_true, g.sp_hat, s=4, alpha=0.35, color=c, label=sc)
    axes[0].plot([0, 30], [0, 30], "k-", lw=0.8)
    axes[0].set_xlim(0, 25)
    axes[0].set_ylim(0, 25)
    axes[0].set_xlabel("true S&P (mm/d)")
    axes[0].set_ylabel("estimated S&P (mm/d)")
    axes[0].legend(frameon=False, fontsize=8, markerscale=3)
    axes[0].set_title("Synthetic twin: per-field S&P recovery")

    x = np.arange(len(scen))
    axes[1].bar(x - 0.27, [summ[s]["next_dry_err_days_et_only"] for s in scen], 0.27, label="ET-only advisory", color="#d62728")
    axes[1].bar(x, [summ[s]["next_dry_err_days_regional_const"] for s in scen], 0.27, label="ET + regional S&P", color="#bbb")
    axes[1].bar(x + 0.27, [summ[s]["next_dry_err_days_field"] for s in scen], 0.27, label="ET + field-learned S&P", color="#1f77b4")
    axes[1].set_xticks(x, [s.split(" ", 1)[0] for s in scen], fontsize=8)
    axes[1].set_ylabel("median error, days until field is dry")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("When will this field need water again?")

    im = axes[2].imshow(sp_map, cmap="viridis_r", vmin=0, vmax=15)
    axes[2].set_title("Real S2, Jamalpur: learned S&P (mm/d), 60 m")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_field_leakage.png", dpi=140)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    df, summ = part_a(rng)
    print(json.dumps(summ, indent=2))
    res_b, sp_map, swir = part_b()
    print(json.dumps(res_b, indent=2))
    (OUT / "leakage_metrics.json").write_text(json.dumps({"A_synthetic_twin": summ, "B_real": res_b}, indent=2))
    plot(df, summ, sp_map, swir)


if __name__ == "__main__":
    main()
