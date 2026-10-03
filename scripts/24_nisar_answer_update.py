"""Experiment 24: a NISAR pass as an answer update to the forecast.

Experiment 23 found that a learned NISAR input adds little, but that under the canopy one NISAR wet/dry call
(06:00) beats the model's own estimate of the field's state that morning. Here the pass is used as Experiment
13 used a farmer's "is there water now?" answer: on each NISAR day the morning forecast of the default model
(v1b, which has never seen NISAR) is re-weighted by Bayes' rule with the pass as a yes/no answer
(kp.answer_likelihood, p_correct = NISAR's assumed accuracy, 0.8). No retraining.

Forecasts issued on NISAR days 29-76 (data before the day; the next-dry horizon stays inside the weather
record), for all in-season fields and for fields under the canopy (true LAI >= 2):
  v1b                 no update;
  v1b + NISAR         the pass as a yes/no answer, assumed accuracy 0.8;
  v1n                 the model retrained with NISAR (Experiment 23), which knows the earlier passes, not
                      this morning's;
  v1b + farmer        a farmer's yes/no answer (right 95 % of the time) on the same mornings, as Experiment 13;
  v1b + NISAR + farmer  both answers.
Test sets: Experiment 23's (training world, random, random + systematic; random with true NISAR accuracy 0.7
and 0.9, still assumed 0.8; 6-day revisit in the two mismatched worlds).
Metrics: next-dry error (median; share off by > 1 day), the irrigate-by-tomorrow call, "ponded now?" after the
update (posterior > 0.5), 90 % coverage.

Outputs: results/nisar_answer_update.json, results/figures/fig23_nisar_answer_update.png.
Needs the Experiment 23 caches and weights. Runtime about 20 minutes (`--plot-only` redraws the figure).
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
m13 = __import__("13_value_of_water_now_answer")
m23 = __import__("23_nisar_input")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
K = 200
ASSUMED_ACC = 0.8
LAST_DAY = 76
VARIANTS = ("v1b", "v1b + NISAR", "v1n", "v1b + farmer", "v1b + NISAR + farmer")


def score(D, W, truth, h_s, h_true, sel):
    m = m13.metrics(D[:, sel], W[:, sel], truth[sel])
    p_wet = (W[:, sel] * (h_s[:, sel] > 0)).sum(0) / W[:, sel].sum(0)
    m["ponded_now_bacc"] = m11.bacc(h_true[sel] > 0, p_wet > 0.5)
    m["n"] = int(sel.sum())
    return m


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    d, d6 = m23.sets(inp), m23.sets6(inp)
    models = {}
    for name, path in (("v1b", OUT / "models" / "kgml_paddy_v1b_pretrained.pt"), ("v1n", m23.V1N)):
        models[name] = kp.PaddyNet()
        models[name].load_state_dict(torch.load(path))
    tests = [(n, d[f"test/{n}"], 12) for n in m23.TESTS] + [(n, d6[f"test/{n}"], 6) for n in m23.TESTS6]
    rows = []
    for name, te, revisit in tests:
        t = time.time()
        for day in [x for x in range(5, LAST_DAY + 1, revisit) if x >= 29]:
            X = te["X"]
            h_true_all = te["H"][:, day - 1]
            F = np.flatnonzero(np.isfinite(h_true_all) & (X[:, day - 1, kp.CH["season"]] > 0)
                               & ((X[:, day, kp.CH["nisar_wet"]] > 0) | (X[:, day, kp.CH["nisar_dry"]] > 0)))
            h_true = h_true_all[F]
            t0 = inp["s0"] + day
            truth = kp.forecast_dry(te["theta"][F], h_true, t0, te["tp_abs"][F], inp, et_f=te["et_f"][F], g=te["g"][F])
            canopy = ps.lai_curve(t0, te["tp_abs"][F], te["g"][F]) >= 2
            nisar_said = (X[F, day, kp.CH["nisar_wet"]] > 0).astype(int)
            farmer_said = m13.simulate_answers(h_true, "yesno", np.random.default_rng(1000 + day))
            forecasts = {}
            for mname, Xm in (("v1b", m23.no_nisar(X)), ("v1n", X)):
                th, st = kp.predict_v1(models[mname], Xm, te["static"], day, K)
                h_s = kp.sample_depth(st, day - 1, K, np.random.default_rng(day), fields=F)
                forecasts[mname] = (h_s, kp.forecast_dry(th[:, F], h_s, t0, te["tp_abs"][F][None, :], inp))
            h_s, D = forecasts["v1b"]
            Ln = kp.answer_likelihood(h_s, nisar_said, "yesno", p_correct=ASSUMED_ACC)
            Lf = kp.answer_likelihood(h_s, farmer_said, "yesno")
            weights = {"v1b": np.ones_like(D), "v1b + NISAR": Ln, "v1b + farmer": Lf, "v1b + NISAR + farmer": Ln * Lf}
            for group, sel in (("all fields", np.ones(len(F), bool)), ("canopy (LAI >= 2)", canopy)):
                for v, W in weights.items():
                    rows.append({"test": name, "day": day, "group": group, "variant": v, **score(D, W, truth, h_s, h_true, sel)})
                hn, Dn = forecasts["v1n"]
                rows.append({"test": name, "day": day, "group": group, "variant": "v1n", **score(Dn, np.ones_like(Dn), truth, hn, h_true, sel)})
        print(f"{name}: done ({time.time() - t:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    cols = ["next_dry_err_days", "share_err_gt1d", "irrigate_by_tomorrow_bacc", "ponded_now_bacc", "ci90_coverage"]
    agg = df.groupby(["test", "group", "variant"])[cols].mean()
    print(agg.round(3).to_string())
    summary = {f"{t_}|{g}|{v}": {c: round(float(x), 4) for c, x in r.items()} for (t_, g, v), r in agg.iterrows()}
    (OUT / "nisar_answer_update.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    tests = list(dict.fromkeys(k.split("|")[0] for k in summary))
    colors = {"v1b": "#9ec5f0", "v1b + NISAR": "#8b5cf6", "v1n": "#d9cdfa", "v1b + farmer": "#e8a33d", "v1b + NISAR + farmer": "#4a3aa7"}
    labels = {"v1b": "v1b, no update", "v1b + NISAR": "v1b + NISAR pass as an answer", "v1n": "v1n (NISAR as a learned input)",
              "v1b + farmer": "v1b + farmer's yes/no", "v1b + NISAR + farmer": "v1b + NISAR + farmer"}
    panels = [("all fields", "next_dry_err_days", "Next-dry error (days), all fields"),
              ("canopy (LAI >= 2)", "next_dry_err_days", "Next-dry error (days), LAI ≥ 2"),
              ("canopy (LAI >= 2)", "irrigate_by_tomorrow_bacc", "Irrigate by tomorrow? LAI ≥ 2"),
              ("canopy (LAI >= 2)", "ponded_now_bacc", "Ponded now? LAI ≥ 2")]
    y = np.arange(len(tests))
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5.4), sharey=True)
    for ax, (g, m, title) in zip(axes, panels):
        for j, v in enumerate(VARIANTS):
            ax.barh(y + (j - 2) * 0.16, [summary[f"{t}|{g}|{v}"][m] for t in tests], height=0.15, color=colors[v], label=labels[v])
        if "bacc" in m:
            ax.set_xlim(0.5, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace(", ", ",\n") for t in tests], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("A NISAR pass as an answer to \"is there water now?\" (forecasts on NISAR mornings; NISAR assumed 0.8 accurate)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "figures" / "fig23_nisar_answer_update.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "nisar_answer_update.json").read_text()))
    else:
        main()
