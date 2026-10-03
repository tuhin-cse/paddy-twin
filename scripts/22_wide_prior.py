"""Experiment 22: re-pre-train KGML-Paddy on a wider S&P prior.

Experiment 20 found that no variant extrapolates to soils leakier than pre-training covered: base S&P
is lognormal with median 5 mm/d and log-sd 0.6 in the prior world, so a district with a median of
10 mm/d is mostly outside it. Here v1b's recipe (Experiment 17: p_nolog 0.3, S&P law, physics loss,
30 epochs, 20,000 fields) is re-run on priors with a log-sd of 0.8 and 1.0 (same median), and compared
with v1b (0.6) on:
  - the Experiment 11 test sets (training world, random mismatch, random + systematic): the cost, if
    any, of spreading the training fields over a wider range;
  - districts outside the old range: leaky (median 10 mm/d), tight (median 2 mm/d), and the
    Experiment 20 deep-ponding and hot-season shifts;
  - the leaky district after fine-tuning on 20 of its logger fields (3 draws).
Metrics as Experiment 20: season S&P next-dry error, log bias, 90 % interval coverage; forecast
next-dry error and coverage.

Outputs: results/wide_prior.json, results/figures/fig21_wide_prior.png, and the weights
results/models/kgml_paddy_v1c_pretrained.pt (log-sd 0.8) and results/models/ablation/wide10.pt (1.0).
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m11 = __import__("11_kgml_paddy_v1_state")
m17 = __import__("17_train_with_nonloggers")
m20 = __import__("20_sp_law_extrapolation")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "wide_prior_sets.pkl"
PRIOR = ps.WORLDS["prior"]
SIGMAS = {"v1b": 0.6, "v1c": 0.8, "wide10": 1.0}
WEIGHTS = {"v1b": OUT / "models" / "kgml_paddy_v1b_pretrained.pt",
           "v1c": OUT / "models" / "kgml_paddy_v1c_pretrained.pt",
           "wide10": OUT / "models" / "ablation" / "wide10.pt"}
TIGHT_TEST = replace(PRIOR, a_median=2.0, a_sigma=0.4)
FT_SEEDS = (91, 92, 93)
N_FT = 20


def sets(inp):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {}
    for name, sig in SIGMAS.items():
        if name == "v1b":
            continue
        cfg = replace(PRIOR, a_sigma=sig)
        out[f"train/{name}"] = ps.simulate(cfg, 20_000, 101 + int(10 * sig), inp)
        out[f"val/{name}"] = ps.simulate(cfg, 2_000, 201 + int(10 * sig), inp)
    out["test/tight"] = ps.simulate(TIGHT_TEST, 2_000, 301, inp)
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        out[k] = {kk: v for kk, v in d.items() if kk not in ("logger", "h6")}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def model(name, d):
    path = WEIGHTS[name]
    m = kp.PaddyNet()
    if not path.exists():
        t = time.time()
        m = kp.pretrain_v1(d[f"train/{name}"], d[f"val/{name}"], epochs=30, p_nolog=m17.P_NOLOG, log=lambda _: None)
        torch.save(m.state_dict(), path)
        print(f"  pre-trained {name} (log-sd {SIGMAS[name]}) in {time.time() - t:.0f} s", flush=True)
    m.load_state_dict(torch.load(path))
    return m


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d11, d20, d = m11.sets(inp), m20.sets(inp), sets(inp)
    tests = {"training world": (d11["test/prior"], inp), "random mismatch": (d11["test/reality"], inp),
             "random + systematic": (d11["test/reality_systematic"], inp),
             "leaky district": (d20["test/leaky"], inp), "tight district": (d["test/tight"], inp),
             "deep ponding": (d20["test/deep"], inp),
             "hot, dry season": (d20["test/hot"], m20.shifted_inputs(inp, m20.SHIFTS["hot"][2]))}
    replay = {"v1b": d11["train"], "v1c": d.get("train/v1c"), "wide10": d.get("train/wide10")}
    summary = {"sigmas": SIGMAS, "results": {}}
    for name in SIGMAS:
        m = model(name, d)
        res = {}
        for tname, (te, inp_t) in tests.items():
            res[tname] = m20.evaluate(m, te, inp_t, et_mid)
            o = res[tname]
            print(f"{name} | {tname}: S&P {o['sp_next_dry_err_days']:.2f} d, bias {o['sp_log_bias']:+.2f}, "
                  f"cov {o['sp_ci90_coverage']:.2f} | forecast {o['forecast_next_dry_err_days']:.2f} d, "
                  f"cov {o['forecast_ci90_coverage']:.2f}", flush=True)
        leaky_cfg = m20.SHIFTS["leaky"][1]
        rows = []
        for r_, seed in enumerate(FT_SEEDS):
            lg = m20.few_loggers(leaky_cfg, inp, seed)
            rows.append(m20.evaluate(kp.fine_tune_v1(model(name, d), lg, replay[name], seed=r_), d20["test/leaky"],
                                     inp, et_mid))
        df = pd.DataFrame(rows)
        res["leaky district + 20 loggers"] = {**df.mean().to_dict(), **{f"{k}_sd": v for k, v in df.std().to_dict().items()}}
        o = res["leaky district + 20 loggers"]
        print(f"{name} | leaky + {N_FT} loggers: S&P {o['sp_next_dry_err_days']:.2f} (sd "
              f"{o['sp_next_dry_err_days_sd']:.2f}) d, bias {o['sp_log_bias']:+.2f}, cov {o['sp_ci90_coverage']:.2f}",
              flush=True)
        summary["results"][name] = res
    (OUT / "wide_prior.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    names = list(summary["sigmas"])
    tests = list(summary["results"][names[0]])
    panels = [("sp_next_dry_err_days", "Season S&P: next-dry error (days)"),
              ("sp_ci90_coverage", "S&P 90 % interval coverage"),
              ("forecast_next_dry_err_days", "Forecast next-dry error (days)")]
    colors = {"v1b": "#9ec5f0", "v1c": "#2a78d6", "wide10": "#0b3d7a"}
    y = np.arange(len(tests))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        for j, n in enumerate(names):
            vals = [summary["results"][n][t][m] for t in tests]
            ax.barh(y + (j - 1) * 0.27, vals, height=0.25, color=colors[n],
                    label=f"{n} (S&P prior log-sd {summary['sigmas'][n]})")
        if "coverage" in m:
            ax.axvline(0.9, color="#52514e", lw=1, ls="--")
            ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace(" + ", "\n+ ") for t in tests], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Pre-training on a wider S&P prior: KGML-Paddy v1b (log-sd 0.6) vs 0.8 and 1.0", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "figures" / "fig21_wide_prior.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "wide_prior.json").read_text()))
    else:
        main()
