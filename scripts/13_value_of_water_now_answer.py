"""Experiment 13: what is one "is there water in your field now?" answer worth?

Experiments 11-12 found that today's water state, not the S&P, limits the next-dry forecast, and
that loggers on other fields cannot supply it. Here the farmer supplies it: on the forecast morning
the app asks "is there water in the field now?", and the answer updates the model's belief about
today's level by Bayes' rule (kp.answer_likelihood), with no retraining. That is what the app can do.

Answers (simulated from the true level, with errors):
  yesno  water / no water; the farmer is right 95 % of the time;
  depth  no water / a little (< 20 mm) / some (20-50) / a lot (> 50); water or not right 95 % of the
         time, the category right 80 % of the time and otherwise one step off;
  gauge  a staff-gauge reading, +/- 5 mm (upper bound: what a cheap marked stick in the field gives).
Who is asked: the fraction f of fields where the forecast is most uncertain (sd of the next-dry
samples), or f of fields at random.

For all in-season test fields (Experiment 11 test sets, pre-trained v1, forecasts from days 45, 55
and 65 averaged), including fields already dry (true answer 0 days):
  next-dry error (median, and share off by > 1 day);
  the call the farmer acts on, "irrigate by tomorrow?" (true next-dry <= 1 day), balanced accuracy;
  90 % interval coverage.
Forecast samples are computed once per world and day; answers only re-weight them.

Outputs: results/value_of_water_now.json, results/figures/fig12_value_of_water_now.png.
Needs the Experiment 11 cache and weights. Runtime about 10 minutes.
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
m11 = __import__("11_kgml_paddy_v1_state")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = ("reality", "reality_systematic")
DAYS = m11.FORECAST_DAYS
K = 400
FRACS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
KINDS = ("yesno", "depth", "gauge")
CH = kp.CH


def simulate_answers(h, kind, rng, p_correct=0.95, cat_correct=0.8, gauge_sd=5.0):
    wet = h > 0
    said_wet = np.where(rng.random(h.size) < p_correct, wet, ~wet)
    if kind == "yesno":
        return said_wet.astype(int)
    if kind == "depth":
        c = kp.depth_category(np.where(wet, h, 1.0))  # category of the water that is there
        step = np.where(rng.random(h.size) < cat_correct, 0, rng.choice([-1, 1], h.size))
        c = np.clip(c + step, 1, 3)
        return np.where(said_wet, c, 0)
    if kind == "gauge":
        return np.where(wet, np.maximum(h + gauge_sd * rng.standard_normal(h.size), 0.0), 0.0)
    raise ValueError(kind)


def metrics(D, W, truth):
    lo, med, hi = kp.weighted_quantiles(D, W, (0.05, 0.5, 0.95))
    err = np.abs(med - truth)
    call_true, call_pred = truth <= 1, med <= 1
    return {"next_dry_err_days": float(np.median(err)), "share_err_gt1d": float((err > 1).mean()),
            "irrigate_by_tomorrow_bacc": m11.bacc(call_true, call_pred),
            "ci90_coverage": float(((lo <= truth + 0.05) & (truth - 0.05 <= hi)).mean())}


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    d = m11.sets(inp)
    model = kp.PaddyNet()
    model.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1_pretrained.pt"))
    rows = []
    for w in WORLDS:
        te = d[f"test/{w}"]
        for day in DAYS:
            t = time.time()
            th, st = kp.predict_v1(model, te["X"], te["static"], day, K)
            h_true_all = te["H"][:, day - 1]
            F = np.flatnonzero(np.isfinite(h_true_all) & (te["X"][:, day - 1, CH["season"]] > 0))
            rng = np.random.default_rng(day)
            h_s = kp.sample_depth(st, day - 1, K, rng, fields=F)
            t0 = inp["s0"] + day
            D = kp.forecast_dry(th[:, F], h_s, t0, te["tp_abs"][F][None, :], inp)
            h_true = h_true_all[F]
            truth = kp.forecast_dry(te["theta"][F], h_true, t0, te["tp_abs"][F], inp, et_f=te["et_f"][F], g=te["g"][F])
            sd = D.std(0)
            base = {"world": w, "day": day, "n_fields": len(F), "share_ponded": float((h_true > 0).mean())}
            m0 = metrics(D, np.ones_like(D), truth)
            rows.append({**base, "kind": "none", "policy": "none", "frac": 0.0, **m0})
            for kind in KINDS:
                ans = simulate_answers(h_true, kind, np.random.default_rng(1000 + day))
                L = kp.answer_likelihood(h_s, ans, kind)
                for frac in FRACS[1:]:
                    n_ask = int(round(frac * len(F)))
                    targeted = np.zeros(len(F), bool)
                    targeted[np.argsort(-sd)[:n_ask]] = True
                    rows.append({**base, "kind": kind, "policy": "targeted", "frac": frac,
                                 **metrics(D, np.where(targeted[None, :], L, 1.0), truth)})
                    if frac < 1.0:
                        rs = []
                        for s in range(3):
                            rnd = np.zeros(len(F), bool)
                            rnd[np.random.default_rng(s).choice(len(F), n_ask, replace=False)] = True
                            rs.append(metrics(D, np.where(rnd[None, :], L, 1.0), truth))
                        rows.append({**base, "kind": kind, "policy": "random", "frac": frac,
                                     **pd.DataFrame(rs).mean().to_dict()})
            print(f"{w} day {day}: {len(F)} fields, {base['share_ponded']:.0%} ponded, no answer "
                  f"{m0['next_dry_err_days']:.2f} d  ({time.time() - t:.0f} s)",
                  flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby(["world", "kind", "policy", "frac"]).mean(numeric_only=True).drop(columns=["day"])
    summary = {f"{w}|{k}|{p}|{f}": {c: round(float(v), 3) for c, v in r.items()} for (w, k, p, f), r in agg.iterrows()}
    (OUT / "value_of_water_now.json").write_text(json.dumps(summary, indent=2))
    print(agg[["next_dry_err_days", "share_err_gt1d", "irrigate_by_tomorrow_bacc", "ci90_coverage"]].round(3).to_string())
    plot(summary)


def plot(summary):
    get = lambda w, k, p, f, m: summary[f"{w}|{k}|{p}|{f}"][m]  # noqa: E731
    series = {("yesno", "targeted"): ("Yes/no, ask the most uncertain", "#2a78d6", "-"),
              ("yesno", "random"): ("Yes/no, ask at random", "#2a78d6", ":"),
              ("depth", "targeted"): ("Yes/no + rough depth, most uncertain", "#eb6834", "-"),
              ("gauge", "targeted"): ("Gauge reading, most uncertain (upper bound)", "#1baf7a", "-")}
    titles = {"reality": "Random mismatch", "reality_systematic": "Random + systematic mismatch"}
    metrics_ = [("next_dry_err_days", "Next-dry error, median (days)"),
                ("irrigate_by_tomorrow_bacc", "Irrigate-by-tomorrow call\n(balanced accuracy)")]
    fig, axes = plt.subplots(len(WORLDS), 2, figsize=(11, 7), sharex=True)
    for r, w in enumerate(WORLDS):
        for c, (m, ylab) in enumerate(metrics_):
            ax = axes[r, c]
            y0 = get(w, "none", "none", 0.0, m)
            for (k, p), (label, col, ls) in series.items():
                fr = [f for f in FRACS[1:] if f"{w}|{k}|{p}|{f}" in summary]
                ax.plot([0.0] + fr, [y0] + [get(w, k, p, f, m) for f in fr], marker="o", ms=4, lw=2, color=col, ls=ls,
                        label=label)
            ax.set_ylabel(f"{titles[w]}\n{ylab}" if c == 0 else ylab)
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == len(WORLDS) - 1:
                ax.set_xlabel("Share of fields asked on the forecast morning")
            ticks = (0.0, 0.2, 0.5, 0.75, 1.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(f * 100)} %" for f in ticks])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Value of a farmer\'s "is there water now?" answer (KGML-Paddy v1, Bayesian update, no retraining)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "figures" / "fig12_value_of_water_now.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "value_of_water_now.json").read_text()))
    else:
        main()
