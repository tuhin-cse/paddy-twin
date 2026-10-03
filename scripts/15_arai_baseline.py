"""Experiment 15: Arai-style baseline against KGML-Paddy v1.

The closest prior method (Arai et al. 2022) keeps the water-loss law constant over the region and
estimates each field's water level and the farmer's irrigation threshold by assimilating satellite
submergence with a particle filter (awdrt.arai_baseline). Here it assimilates the same Sentinel-2 and
Sentinel-1 passes as KGML-Paddy, with the regional loss law fitted on 80 logger fields (medians of the
per-logger fits), in two variants:
  arai        satellite only, as Arai et al.;
  arai_logs   plus the farmer's logged irrigations applied as known events;
against KGML-Paddy v1 as pre-trained (Experiment 11). Same test fields (Experiment 11 sets), forecasts
from days 45, 55, 65 using data before that day only:
  ponded today (causal balanced accuracy);
  next-dry error for all in-season fields and for ponded fields, 90 % coverage;
  the "irrigate by tomorrow?" call (balanced accuracy);
  season-end effective S&P (Arai: the regional constant);
  Arai's own product, the farmer's threshold (Spearman with the true farmer type threshold).

Outputs: results/arai_baseline.json, results/figures/fig14_arai_baseline.png. Runtime about 15 minutes.
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
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m9 = __import__("09_kgml_paddy_v0")
m11 = __import__("11_kgml_paddy_v1_state")
from awdrt import arai_baseline as ab  # noqa: E402
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = ("prior", "reality", "reality_systematic")
LOGGER_SEEDS = {"prior": 31, "reality": 32, "reality_systematic": 33}
DAYS = m11.FORECAST_DAYS
K = 200
CH = kp.CH
METHODS = ("kgml", "arai", "arai_logs")


def forecast_metrics(Dsamp, p_wet, truth, h_true):
    """Dsamp (k, n) next-dry samples; p_wet (n,) P(ponded today); truth (n,) true next-dry; h_true (n,)."""
    lo, med, hi = np.quantile(Dsamp, 0.05, 0), np.median(Dsamp, 0), np.quantile(Dsamp, 0.95, 0)
    err = np.abs(med - truth)
    cov = (lo <= truth + 0.05) & (truth - 0.05 <= hi)
    ponded = h_true > 10
    return {"ponded_today_bacc": m11.bacc(h_true > 0, p_wet > 0.5),
            "next_dry_err_all_days": float(np.median(err)), "share_err_gt1d_all": float((err > 1).mean()),
            "irrigate_by_tomorrow_bacc": m11.bacc(truth <= 1, med <= 1), "ci90_coverage_all": float(cov.mean()),
            "next_dry_err_ponded_days": float(np.median(err[ponded])), "ci90_coverage_ponded": float(cov[ponded].mean())}


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    model = kp.PaddyNet()
    model.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1_pretrained.pt"))
    summary = {}
    for w in WORLDS:
        te = d[f"test/{w}"]
        lg = ps.simulate(ps.WORLDS[w], 80, LOGGER_SEEDS[w], inp)
        lg.update(zip(("h", "tau", "r"), ps.recession_rates(lg["logger"], lg["tp_abs"], inp)))
        th_reg = ab.regional_theta(lg)
        print(f"{w}: regional loss law a={th_reg[0]:.2f} mm/d, b={th_reg[1]:.3f} /d", flush=True)
        t = time.time()
        pf = {m: ab.run(te, inp, th_reg, set(DAYS), use_logs=(m == "arai_logs"), P=K, seed=1) for m in ("arai", "arai_logs")}
        print(f"  particle filters done ({time.time() - t:.0f} s)", flush=True)
        rows = []
        for day in DAYS:
            H = te["H"][:, day - 1]
            F = np.flatnonzero(np.isfinite(H) & (te["X"][:, day - 1, CH["season"]] > 0))
            t0 = inp["s0"] + day
            tp = te["tp_abs"][F]
            truth = kp.forecast_dry(te["theta"][F], H[F], t0, tp, inp, et_f=te["et_f"][F], g=te["g"][F])
            th, st = kp.predict_v1(model, te["X"], te["static"], day, K)
            h_s = kp.sample_depth(st, day - 1, K, np.random.default_rng(day), fields=F)
            res = {"kgml": forecast_metrics(kp.forecast_dry(th[:, F], h_s, t0, tp[None, :], inp),
                                            st["p_wet"][F, day - 1], truth, H[F])}
            # decomposition: which matters for the next-dry forecast, the state or the per-field law?
            reg = np.broadcast_to(th_reg, (K, len(F), 4))
            true_h = np.broadcast_to(H[F], (K, len(F)))
            res["kgml_state_regional_law"] = forecast_metrics(kp.forecast_dry(reg, h_s, t0, tp[None, :], inp),
                                                              st["p_wet"][F, day - 1], truth, H[F])
            res["true_state_kgml_law"] = forecast_metrics(kp.forecast_dry(th[:, F], true_h, t0, tp[None, :], inp),
                                                          (H[F] > 0).astype(float), truth, H[F])
            res["true_state_regional_law"] = forecast_metrics(kp.forecast_dry(reg[:1], H[F][None, :], t0, tp[None, :], inp),
                                                              (H[F] > 0).astype(float), truth, H[F])
            for m in ("arai", "arai_logs"):
                hp = pf[m][0][day]["h"][F].T  # (P, n)
                Dp = kp.forecast_dry(np.broadcast_to(th_reg, (*hp.shape, 4)), np.nan_to_num(hp, nan=0.0), t0, tp[None, :], inp)
                res[m] = forecast_metrics(Dp, np.nanmean(hp > 0, 0), truth, H[F])
            for m, r in res.items():
                rows.append({"method": m, "day": day, **r})
        df = pd.DataFrame(rows).groupby("method").mean(numeric_only=True).drop(columns="day")
        out = {m: {k: round(float(v), 3) for k, v in df.loc[m].items()} for m in df.index}
        # season-end effective S&P: KGML posterior vs the regional constant
        th_full, _ = kp.predict_v1(model, te["X"], te["static"], te["X"].shape[1], 100)
        out["kgml"]["season_sp"] = m9.evaluate(th_full, te, inp, et_mid)
        sp_reg = ps.sp_effective(*th_reg, te["tp_abs"], inp, dt=0.05)
        out["arai"]["season_sp"] = out["arai_logs"]["season_sp"] = m9.point_metrics(sp_reg, te["sp_eff"], et_mid)
        # Arai's own product: the farmer's threshold
        thr_true = np.select([te["ftype"] == 0, te["ftype"] == 1, te["ftype"] == 2], [0.0, -150.0, 12.5])
        for m in ("arai", "arai_logs"):
            D_mean = pf[m][1].mean(1)
            out[m]["threshold_spearman"] = float(spearmanr(D_mean, thr_true).correlation)
            out[m]["awd_farmer_detection_bacc"] = m11.bacc(te["ftype"] == 1, D_mean < -75)
        out["regional_theta"] = th_reg.round(4).tolist()
        summary[w] = out
        print(f"== {w}")
        for m in ("kgml_state_regional_law", "true_state_kgml_law", "true_state_regional_law"):
            print(f"  {m:24s} next-dry all {out[m]['next_dry_err_all_days']:.2f} d, ponded {out[m]['next_dry_err_ponded_days']:.2f} d, "
                  f"irrigate-by-tomorrow {out[m]['irrigate_by_tomorrow_bacc']:.3f}")
        for m in METHODS:
            o = out[m]
            print(f"  {m:10s} ponded-today {o['ponded_today_bacc']:.3f} | next-dry all {o['next_dry_err_all_days']:.2f} d, "
                  f"ponded {o['next_dry_err_ponded_days']:.2f} d | irrigate-by-tomorrow {o['irrigate_by_tomorrow_bacc']:.3f} | "
                  f"cov {o['ci90_coverage_all']:.2f} | season S&P {o['season_sp']['next_dry_err_days']:.2f} d, "
                  f"rho {o['season_sp']['spearman']:.2f}" +
                  (f" | threshold rho {o['threshold_spearman']:.2f}, AWD detection {o['awd_farmer_detection_bacc']:.2f}"
                   if m != "kgml" else ""), flush=True)
    (OUT / "arai_baseline.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    colors = {"kgml": ("#2a78d6", "o", "KGML-Paddy v1"), "arai": ("#8a8983", "s", "Arai-style, satellite only"),
              "arai_logs": ("#eb6834", "D", "Arai-style + farmer logs")}
    titles = {"prior": "Training\nworld", "reality": "Random\nmismatch", "reality_systematic": "Random +\nsystematic"}
    panels = [("ponded_today_bacc", "Ponded today?\n(balanced accuracy)"),
              ("next_dry_err_all_days", "Next-dry error, all fields\n(median, days)"),
              ("irrigate_by_tomorrow_bacc", "Irrigate-by-tomorrow call\n(balanced accuracy)"),
              ("season_sp", "Season S&P: next-dry error\n(median, days)")]
    x = np.arange(len(WORLDS))
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.9))
    for ax, (key, lab) in zip(axes, panels):
        for j, m in enumerate(METHODS):
            col, mk, name = colors[m]
            ys = [summary[w][m]["season_sp"]["next_dry_err_days"] if key == "season_sp" else summary[w][m][key]
                  for w in WORLDS]
            ax.scatter(x + (j - 1) * 0.18, ys, color=col, marker=mk, s=45, label=name, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([titles[w] for w in WORLDS], fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("KGML-Paddy v1 against an Arai-style particle filter with a regional loss law (same passes, same fields)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "figures" / "fig14_arai_baseline.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "arai_baseline.json").read_text()))
    else:
        main()
