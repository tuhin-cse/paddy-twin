"""Experiment 11: KGML-Paddy v1, the daily water depth and the next-dry forecast.

v1 (awdrt.kgml_paddy.PaddyNet) adds to v0 a daily state head: for the end of each day, the
probability that the field is ponded, the ponded depth and the depth below the surface. It is
trained on the simulator's true water levels, tied to the S&P law by a water-balance consistency
loss, and trained with observations hidden after a random day, so it can forecast from "today".

Same fields as Experiment 9 (seeds 1, 2, 3, 5, 7), so the season-end S&P can be compared with v0.
Evaluated:
  A. season-end effective S&P (as Experiments 9-10);
  B. the daily depth: error, and wet/dry accuracy on satellite days by LAI band, compared with
     the raw Sentinel-2 / Sentinel-1 detections (a first look at N3; it uses the whole season);
  C. the next-dry forecast from days 45, 55 and 65 of the season (29 Feb, 10 and 20 Mar), using
     only the data before that day, for fields ponded then: days until the water is gone if the
     farmer does not irrigate. The error is split into state and S&P parts with oracle variants,
     and compared with a regional S&P and with ET only (what IRAS-type advice assumes).

Truth for C integrates the true depth with the true S&P law and the field's true ET0 factor and
crop growth, without lateral exchange and without further irrigation.

Outputs: results/kgml_paddy_v1.json, results/models/kgml_paddy_v1_pretrained.pt,
results/figures/fig10_kgml_paddy_v1_state.png. Runtime about 15 minutes.
"""

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m9 = __import__("09_kgml_paddy_v0")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "kgml_v1_sets.pkl"
TESTS = {"prior": ("prior", 3), "reality": ("reality", 5), "reality_systematic": ("reality_systematic", 7)}
FORECAST_DAYS = (45, 55, 65)
N_SAMPLES = 100
CH = kp.CH


def sets(inp):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {"train": ps.simulate(ps.WORLDS["prior"], 20_000, 1, inp), "val": ps.simulate(ps.WORLDS["prior"], 2_000, 2, inp)}
    for name, (w, seed) in TESTS.items():
        out[f"test/{name}"] = ps.simulate(ps.WORLDS[w], 2_000, seed, inp)
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        drop = ("logger", "h6") if k in ("train", "val") else ("logger",)
        out[k] = {kk: v for kk, v in d.items() if kk not in drop}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def bacc(truth, pred):
    t, p = truth.astype(bool), pred.astype(bool)
    if t.all() or (~t).all():
        return float("nan")
    return float(0.5 * ((t & p).sum() / t.sum() + (~t & ~p).sum() / (~t).sum()))


def depth_metrics(d, st, inp):
    """Daily water-level errors, and wet/dry balanced accuracy on satellite days by LAI band."""
    H, X, h6 = d["H"], d["X"], d["h6"]
    season = X[:, :, CH["season"]] > 0
    ok = season & np.isfinite(H)
    wet = ok & (H > 0)
    both = wet & (st["p_wet"] > 0.5)
    out = {"ponded_bacc_all_days": bacc(H[ok] > 0, st["p_wet"][ok] > 0.5),
           "ponded_depth_mae_mm": float(np.mean(np.abs(st["pos_mu"][both] - H[both]))),
           "below_surface_mae_mm": float(np.mean(np.abs(st["neg_mu"][ok & (H <= 0) & (st["p_wet"] <= 0.5)]
                                                        - H[ok & (H <= 0) & (st["p_wet"] <= 0.5)])))}
    lai = X[:, :, CH["lai_obs"]] * 5
    T = X.shape[1]
    # P(ponded) at noon (S2) and 18:00 (S1): between the end of the previous day and of this day
    pw = st["p_wet"]
    prev = np.concatenate([pw[:, :1], pw[:, :-1]], 1)
    est_noon, est_eve = 0.5 * (prev + pw) - 0.5, 0.25 * prev + 0.75 * pw - 0.5
    for sensor, q, est in (("s2", 2, est_noon), ("s1", 3, est_eve)):
        w, dr = X[:, :, CH[f"{sensor}_wet"]] > 0, X[:, :, CH[f"{sensor}_dry"]] > 0
        has = (w | dr) & season
        truth = h6[:, np.arange(T) * ps.PER_DAY + q] > 0
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 9)):
            band = has & (lai >= lo) & (lai < hi)
            out[f"{sensor}_raw_bacc_lai{lo}-{hi}"] = bacc(truth[band], w[band])
            out[f"{sensor}_state_bacc_lai{lo}-{hi}"] = bacc(truth[band], est[band] > 0)
            out[f"{sensor}_n_lai{lo}-{hi}"] = int(band.sum())
    return out


def forecast_metrics(model, d, inp, day):
    s0 = inp["s0"]
    th, st = kp.predict_v1(model, d["X"], d["static"], day, N_SAMPLES)
    h_true = d["H"][:, day - 1]
    known = np.isfinite(h_true) & (d["X"][:, day - 1, CH["season"]] > 0)
    ponded_today_bacc = bacc(h_true[known] > 0, st["p_wet"][known, day - 1] > 0.5)
    sel = np.isfinite(h_true) & (h_true > 10) & (d["X"][:, day - 1, CH["season"]] > 0)
    t0 = s0 + day
    tp = d["tp_abs"][sel]
    truth = kp.forecast_dry(d["theta"][sel], h_true[sel], t0, tp, inp, et_f=d["et_f"][sel], g=d["g"][sel])
    rng = np.random.default_rng(day)
    h_s = kp.sample_depth(st, day - 1, N_SAMPLES, rng, fields=np.flatnonzero(sel))
    th_s = th[:, sel]
    k = th_s.shape[0]
    sp_reg = np.array([5.64, 0, 0, 0])  # prior median effective S&P (Experiment 9)
    variants = {
        "kgml": (th_s, h_s),
        "kgml_sp_true_depth": (th_s, np.broadcast_to(h_true[sel], (k, sel.sum()))),
        "true_sp_kgml_depth": (np.broadcast_to(d["theta"][sel], (k, sel.sum(), 4)), h_s),
        "regional_sp_kgml_depth": (np.broadcast_to(sp_reg, (k, sel.sum(), 4)), h_s),
        "et_only_kgml_depth": (np.zeros((k, sel.sum(), 4)), h_s),
    }
    out = {"n_fields_ponded": int(sel.sum()), "true_days_median": float(np.median(truth)),
           "ponded_today_bacc_causal": ponded_today_bacc}
    for name, (tt, hh) in variants.items():
        D = kp.forecast_dry(tt, hh, t0, tp[None, :], inp)
        med = np.median(D, 0)
        err = np.abs(med - truth)
        out[name] = {"median_abs_err_days": float(np.median(err)), "share_err_gt1d": float((err > 1).mean()),
                     "bias_days": float(np.median(med - truth))}
        if name == "kgml":
            lo, hi = np.quantile(D, 0.05, 0), np.quantile(D, 0.95, 0)
            out[name]["ci90_coverage"] = float(((lo <= truth + 0.05) & (truth - 0.05 <= hi)).mean())
    return out


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    t = time.time()
    d = sets(inp)
    print(f"sets ready in {time.time() - t:.0f} s", flush=True)
    wpath = OUT / "models" / "kgml_paddy_v1_pretrained.pt"
    if wpath.exists():
        model = kp.PaddyNet()
        model.load_state_dict(torch.load(wpath))
    else:
        t = time.time()
        model = kp.pretrain_v1(d["train"], d["val"], epochs=30, log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), wpath)
        print(f"pre-trained in {time.time() - t:.0f} s", flush=True)

    v0 = json.loads((OUT / "kgml_paddy_v0.json").read_text())
    T = d["train"]["X"].shape[1]
    summary = {}
    for name in TESTS:
        te = d[f"test/{name}"]
        th, st = kp.predict_v1(model, te["X"], te["static"], T, N_SAMPLES)
        res = {"season_end_sp": m9.evaluate(th, te, inp, et_mid), "depth": depth_metrics(te, st, inp),
               "forecast": {int(day): forecast_metrics(model, te, inp, day) for day in FORECAST_DAYS}}
        v0_key = "pretrained_in_distribution" if name == "prior" else None
        res["season_end_sp_v0"] = v0[v0_key] if v0_key else v0["by_world_model_loggers"][f"{name}|pretrained|0"]
        summary[name] = res
        s, s0_ = res["season_end_sp"], res["season_end_sp_v0"]
        print(f"\n== {name}: season-end S&P  v1 {s['next_dry_err_days']:.2f} d / rho {s['spearman']:.2f} / cov "
              f"{s['ci90_coverage']:.2f}   (v0 {s0_['next_dry_err_days']:.2f} / {s0_['spearman']:.2f} / {s0_['ci90_coverage']:.2f})")
        dm = res["depth"]
        print(f"   ponded or not (all days) bacc {dm['ponded_bacc_all_days']:.3f}; depth MAE ponded "
              f"{dm['ponded_depth_mae_mm']:.1f} mm, below surface {dm['below_surface_mae_mm']:.1f} mm")
        for sensor in ("s2", "s1"):
            print("   " + sensor + " bacc raw/state by LAI: " + ", ".join(
                f"{b}: {dm[f'{sensor}_raw_bacc_lai{b}']:.2f}/{dm[f'{sensor}_state_bacc_lai{b}']:.2f}"
                for b in ("0-1.5", "1.5-3", "3-9")))
        for day, f in res["forecast"].items():
            print(f"   forecast day {day} ({f['n_fields_ponded']} ponded, true median {f['true_days_median']:.1f} d): " +
                  ", ".join(f"{k} {v['median_abs_err_days']:.2f}" for k, v in f.items() if isinstance(v, dict)) +
                  f"; kgml cov {f['kgml']['ci90_coverage']:.2f}; ponded-today bacc {f['ponded_today_bacc_causal']:.3f}",
                  flush=True)
    (OUT / "kgml_paddy_v1.json").write_text(json.dumps(summary, indent=2))

    # figure: three example fields (reality): true depth, estimated depth, satellite passes, logged irrigation,
    # and the forecast from day 55 against the true time the water would run out
    te = d["test/reality"]
    day = 55
    _, st_full = kp.predict_v1(model, te["X"], te["static"], T, 50)
    th_c, st_c = kp.predict_v1(model, te["X"], te["static"], day, 50)
    wet_f = st_full["p_wet"] > 0.5
    h_est = st_full["h"]
    h_sd = np.where(wet_f, st_full["pos_sd"], st_full["neg_sd"])
    cand = np.flatnonzero((te["H"][:, day - 1] > 20) & (te["ftype"] == 0))
    pick = cand[np.argsort(te["sp_eff"][cand])[[len(cand) // 10, len(cand) // 2, 9 * len(cand) // 10]]]
    days = np.arange(T)
    dates = pd.to_datetime(ps.SEASON_START) + pd.to_timedelta(days + 1, "D")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True)
    for ax, i in zip(axes, pick):
        season = te["X"][i, :, CH["season"]] > 0
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.plot(dates[season], te["H"][i, season], color="#8a8983", lw=2, label="True depth (end of day)")
        ax.plot(dates[season], h_est[i, season], color="#2a78d6", lw=2, label="KGML-Paddy v1 estimate (whole season)")
        ax.fill_between(dates[season], (h_est - 1.645 * h_sd)[i, season], (h_est + 1.645 * h_sd)[i, season],
                        color="#2a78d6", alpha=0.15, lw=0)
        for sensor, y, col, mk in (("s2", 130, "#1baf7a", "o"), ("s1", 118, "#4a3aa7", "s")):
            for state, filled in (("wet", True), ("dry", False)):
                dd = te["X"][i, :, CH[f"{sensor}_{state}"]] > 0
                ax.scatter(dates[dd], np.full(dd.sum(), y), marker=mk, s=22, color=col if filled else "none",
                           edgecolors=col, label=f"{sensor.upper()} {'wet' if filled else 'dry'}")
        irr = te["X"][i, :, CH["irr_logged"]] > 0
        ax.scatter(dates[irr], np.full(irr.sum(), 105), marker="v", s=26, color="#eb6834", label="Logged irrigation")
        D = kp.forecast_dry(th_c[:, i], kp.sample_depth(st_c, day - 1, 50, np.random.default_rng(i), fields=[i])[:, 0],
                            inp["s0"] + day, te["tp_abs"][i], inp)
        Dt = kp.forecast_dry(te["theta"][i], te["H"][i, day - 1], inp["s0"] + day, te["tp_abs"][i], inp,
                             et_f=te["et_f"][i], g=te["g"][i])
        t0 = pd.to_datetime(ps.SEASON_START) + pd.Timedelta(days=day)
        ax.axvline(t0, color="#52514e", lw=1, ls=":")
        ax.axvspan(t0 + pd.Timedelta(days=float(np.quantile(D, 0.05))), t0 + pd.Timedelta(days=float(np.quantile(D, 0.95))),
                   color="#eda100", alpha=0.25, lw=0, label="Forecast dry date from 10 Mar (90 %)")
        ax.axvline(t0 + pd.Timedelta(days=float(Dt)), color="#e34948", lw=1.5, label="True dry date without irrigation")
        ax.set_ylim(-260, 145)
        ax.set_ylabel("Water level (mm)")
        ax.set_title(f"Field {i}: effective S&P {te['sp_eff'][i]:.1f} mm/d", fontsize=9, loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("KGML-Paddy v1 on the mismatched reality: daily water level and next-dry forecast", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "figures" / "fig10_kgml_paddy_v1_state.png", dpi=150)


if __name__ == "__main__":
    main()
