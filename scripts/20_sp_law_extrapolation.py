"""Experiment 20: does the S&P law help when the test fields lie outside the training range?

Experiment 18 found that the physical structure (the S&P law SP(h, tau) = a(1 + w tau/80)e^(-c tau) + b h)
adds no accuracy in-distribution. Its remaining claim is extrapolation, where a black-box head has
nothing to fall back on. Three shifts, each keeping the simulator's own law (so the structure is
correctly specified, its best case), with KGML-Paddy (S1, P1, L1) with the S&P law (F1) or a black-box
effective-S&P head (F0):

  leaky   train on tight soils (a median 3.5 mm/d, sigma 0.4), test on leaky ones (a median 10):
          most test fields lie outside the training range. Two new pre-trainings.
  deep    train on the prior world (the Experiment 18 weights); test on farmers who pond 100-150 mm
          instead of 40-70 mm, where the depth term b h matters most.
  hot     train on the prior world; test on a hotter, drier season (ET0 x 1.4, rain x 0.3).
For the leaky shift, both variants are also fine-tuned on 20 logger fields from the test world
(transfer to a new district with few loggers; 3 logger draws).

Metrics on 2,000 test fields per shift: season-end effective S&P (next-dry error, log bias, rank
correlation, 90 % interval coverage), forecasts from days 45, 55 and 65 (next-dry error, coverage), and
the forecast for fields deeper than 80 mm at the forecast time.

Outputs: results/sp_law_extrapolation.json, results/figures/fig19_sp_law_extrapolation.png.
Runtime about 40 minutes (`--plot-only` redraws the figure).
"""

import json
import pathlib
import sys
import time
from dataclasses import replace

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
m17 = __import__("17_train_with_nonloggers")
m18 = __import__("18_ablation")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "sp_law_extrapolation_sets.pkl"
PRIOR = ps.WORLDS["prior"]
TIGHT = replace(PRIOR, a_median=3.5, a_sigma=0.4)
SHIFTS = {  # name: (training world, test world, input change)
    "in-distribution": ("prior", PRIOR, None),
    "leaky": ("tight", replace(PRIOR, a_median=10.0, a_sigma=0.4), None),
    "deep": ("prior", replace(PRIOR, h0_median=(100.0, 150.0)), None),
    "hot": ("prior", PRIOR, {"et0": 1.4, "rain": 0.3}),
}
TEST_SEEDS = {"in-distribution": 73, "leaky": 74, "deep": 75, "hot": 76}
FT_SEEDS = (77, 78, 79)
N_FT = 20
K = 100
DAYS = m11.FORECAST_DAYS
DEEP_MM = 80.0


def shifted_inputs(inp, change):
    if change is None:
        return inp
    return {**inp, "et0": inp["et0"] * change["et0"], "rain": inp["rain"] * change["rain"]}


def sets(inp):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {"train/tight": ps.simulate(TIGHT, 20_000, 71, inp), "val/tight": ps.simulate(TIGHT, 2_000, 72, inp)}
    for name, (_, cfg, change) in SHIFTS.items():
        out[f"test/{name}"] = ps.simulate(cfg, 2_000, TEST_SEEDS[name], shifted_inputs(inp, change))
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        out[k] = {kk: v for kk, v in d.items() if kk not in ("logger", "h6")}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def model_for(train, F, d):
    if train == "prior":
        return m18.pretrained(1, F, 1, None)  # Experiment 18 weights (S1-P1-F*-L1)
    path = m18.WDIR / f"tight-S1-P1-F{F}-L1.pt"
    model = kp.PaddyNet(structure=bool(F))
    if not path.exists():
        t = time.time()
        m = kp.pretrain_v1(d["train/tight"], d["val/tight"], epochs=30, p_nolog=m17.P_NOLOG, structure=bool(F),
                           log=lambda _: None)
        torch.save(m.state_dict(), path)
        print(f"  pre-trained tight F{F} in {time.time() - t:.0f} s", flush=True)
    model.load_state_dict(torch.load(path))
    return model


def evaluate(model, te, inp, et_mid):
    X = te["X"]
    th, _ = kp.predict_v1(model, X, te["static"], X.shape[1], K)
    sps = ps.sp_effective(*(th[..., i] for i in range(4)), te["tp_abs"][None, :], inp, dt=0.05)
    sp = np.median(sps, 0)
    y = te["sp_eff"]
    lo, hi = np.quantile(sps, 0.05, 0), np.quantile(sps, 0.95, 0)
    out = {"sp_next_dry_err_days": m9.point_metrics(sp, y, et_mid)["next_dry_err_days"],
           "sp_log_bias": float(np.median(np.log(sp / y))), "sp_spearman": float(spearmanr(sp, y).correlation),
           "sp_ci90_coverage": float(((lo <= y) & (y <= hi)).mean())}
    err, cov, err_deep, n_deep = [], [], [], 0
    for day in DAYS:
        H = te["H"][:, day - 1]
        th_d, st = kp.predict_v1(model, X, te["static"], day, K)
        F = np.flatnonzero(np.isfinite(H) & (X[:, day - 1, kp.CH["season"]] > 0))
        t0, tp = inp["s0"] + day, te["tp_abs"][F]
        truth = kp.forecast_dry(te["theta"][F], H[F], t0, tp, inp, et_f=te["et_f"][F], g=te["g"][F])
        h_s = kp.sample_depth(st, day - 1, K, np.random.default_rng(day), fields=F)
        D = kp.forecast_dry(th_d[:, F], h_s, t0, tp[None, :], inp)
        q5, med, q95 = np.quantile(D, 0.05, 0), np.median(D, 0), np.quantile(D, 0.95, 0)
        e = np.abs(med - truth)
        err.append(e)
        cov.append((q5 <= truth + 0.05) & (truth - 0.05 <= q95))
        deep = H[F] > DEEP_MM
        err_deep.append(e[deep])
        n_deep += int(deep.sum())
    out.update({"forecast_next_dry_err_days": float(np.median(np.concatenate(err))),
                "forecast_ci90_coverage": float(np.concatenate(cov).mean()),
                "forecast_deep_err_days": float(np.median(np.concatenate(err_deep))) if n_deep else float("nan"),
                "n_deep_forecasts": n_deep})
    return out


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = sets(inp)
    m11d = m11.sets(inp)
    summary = {}
    for name, (train, cfg, change) in SHIFTS.items():
        inp_t = shifted_inputs(inp, change)
        te = d[f"test/{name}"]
        res = {}
        for F in (1, 0):
            res[f"F{F}"] = evaluate(model_for(train, F, d), te, inp_t, et_mid)
            o = res[f"F{F}"]
            print(f"{name} F{F}: S&P err {o['sp_next_dry_err_days']:.2f} d, bias {o['sp_log_bias']:+.2f}, "
                  f"rho {o['sp_spearman']:.2f}, cov {o['sp_ci90_coverage']:.2f} | forecast "
                  f"{o['forecast_next_dry_err_days']:.2f} d (deep {o['forecast_deep_err_days']:.2f} d, "
                  f"n={o['n_deep_forecasts']}), cov {o['forecast_ci90_coverage']:.2f}", flush=True)
        if name == "leaky":  # few-shot transfer: fine-tune on 20 logger fields from the test world
            replay = d["train/tight"]
            for F in (1, 0):
                rows = []
                for r_, seed in enumerate(FT_SEEDS):
                    lg = few_loggers(cfg, inp_t, seed)
                    ft = kp.fine_tune_v1(model_for(train, F, d), lg, replay, seed=r_)
                    rows.append(evaluate(ft, te, inp_t, et_mid))
                df = pd.DataFrame(rows)
                res[f"F{F}+ft{N_FT}"] = {**df.mean().to_dict(), **{f"{k}_sd": v for k, v in df.std().to_dict().items()}}
                o = res[f"F{F}+ft{N_FT}"]
                print(f"{name} F{F} + {N_FT} loggers: S&P err {o['sp_next_dry_err_days']:.2f} "
                      f"(sd {o['sp_next_dry_err_days_sd']:.2f}) d, bias {o['sp_log_bias']:+.2f} | forecast "
                      f"{o['forecast_next_dry_err_days']:.2f} d", flush=True)
        res["sp_eff_median_test"] = float(np.median(te["sp_eff"]))
        res["sp_eff_p95_train"] = float(np.quantile((d["train/tight"] if train == "tight" else m11d["train"])["sp_eff"], 0.95))
        res["share_test_above_train_p95"] = float(np.mean(te["sp_eff"] > res["sp_eff_p95_train"]))
        summary[name] = res
    (OUT / "sp_law_extrapolation.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def few_loggers(cfg, inp, seed):
    lg = ps.simulate(cfg, 2 * N_FT, seed, inp)
    lg["H_log"] = ps.end_of_day_depth(lg["logger"])
    lg.update(zip(("h", "tau", "r"), ps.recession_rates(lg["logger"], lg["tp_abs"], inp)))
    lg["sp_log"] = kp.logger_sp_eff(lg, inp)
    keep = np.flatnonzero(np.isfinite(lg["r"]).sum(1) >= 8)[:N_FT]
    n = len(lg["theta"])
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in lg.items()
            if k not in ("logger", "h6")}


def plot(summary):
    groups = [("in-distribution", "F{}", "In\ndistribution"), ("leaky", "F{}", "Leaky\nsoils"),
              ("leaky", "F{}+ft20", "Leaky soils\n+ 20 loggers"), ("deep", "F{}", "Deep\nponding"),
              ("hot", "F{}", "Hot, dry\nseason")]
    panels = [("sp_next_dry_err_days", "Season S&P: next-dry error (days)"),
              ("forecast_next_dry_err_days", "Forecast next-dry error (days)"),
              ("sp_ci90_coverage", "S&P 90 % interval coverage")]
    colors = {1: "#2a78d6", 0: "#eb6834"}
    labels = {1: "S&P law (F1)", 0: "Black-box S&P head (F0)"}
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (m, title) in zip(axes, panels):
        for j, F in enumerate((1, 0)):
            vals = [summary[s][k.format(F)][m] for s, k, _ in groups]
            ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=colors[F], label=labels[F])
            for xi, v in zip(x + (j - 0.5) * 0.38, vals):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
        if "coverage" in m:
            ax.axhline(0.9, color="#52514e", lw=1, ls="--")
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([g[2] for g in groups], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("S&P law vs black-box head outside the training range (KGML-Paddy S1-P1-L1)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig19_sp_law_extrapolation.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "sp_law_extrapolation.json").read_text()))
    else:
        main()
