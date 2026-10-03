"""Experiment 21: regularised fine-tuning on loggers.

Experiment 18b found that fine-tuning KGML-Paddy v1b on 80 logger fields with the Experiment 12 recipe
helps S&P only under systematic mismatch. Under random mismatch it made S&P slightly worse and doubled
its bias (-0.06 -> -0.12 in log terms). Five recipes, applied to v1b (S1-P1-F1-L1):

  default     Experiment 12: 300 steps, head lr 1e-3, encoder lr 2e-4, replay of simulated fields;
  low-lr      learning rates / 3.3 and / 4 (head 3e-4, encoder 5e-5);
  heads-only  encoder frozen (lr 0), heads at 1e-3;
  L2-SP       default plus a penalty anchor_w * sum (p - p0)^2 towards the pre-trained weights;
  early-stop  default on 75 % of the loggers; the other 25 % are held out, and the weights with the
              lowest held-out S&P error (median |log model S&P - log logger S&P|, checked every 25
              steps, including step 0 = no fine-tuning) are kept. This is the only recipe that can
              decide not to fine-tune.

Worlds: random mismatch and random + systematic; 20 and 80 logger fields; 3 logger draws each.
Metrics on the Experiment 11 test fields: season S&P next-dry error and log bias, forecast next-dry
error and 90 % coverage (forecasts from days 45, 55, 65).

Outputs: results/regularised_fine_tuning.json, results/figures/fig20_regularised_fine_tuning.png.
Runtime about 90 minutes (`--threads N`; `--plot-only` redraws the figure).
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
m18b = __import__("18b_ablation_followup")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = {"reality": (81, 82, 83), "reality_systematic": (84, 85, 86)}  # logger-draw seeds
NS = (20, 80)
ANCHOR_W = 0.1  # weight drift of a default fine-tune ~20; anchor 1 cuts it to 0.1 (near-frozen)
RECIPES = {
    "default": {},
    "low-lr": {"lr_head": 3e-4, "lr_enc": 5e-5},
    "heads-only": {"lr_enc": 0.0},
    "L2-SP": {"anchor_w": ANCHOR_W},
    "early-stop": {},
}
K_VAL = 50


def v1b():
    m = kp.PaddyNet()
    m.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1b_pretrained.pt"))
    return m


def subset(lg, idx):
    n = len(lg["theta"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in lg.items()}


def sp_score_fn(hold, inp):
    """Held-out logger S&P error: median |log model S&P - log logger S&P| (the campaign can measure it)."""
    ok = np.isfinite(hold["sp_log"]) & (hold["sp_log"] > 0)
    hold = subset(hold, np.flatnonzero(ok))

    def score(model):
        th, _ = kp.predict_v1(model, hold["X"], hold["static"], hold["X"].shape[1], K_VAL)
        sp = np.median(ps.sp_effective(*(th[..., i] for i in range(4)), hold["tp_abs"][None, :], inp, dt=0.05), 0)
        return float(np.median(np.abs(np.log(sp) - np.log(hold["sp_log"]))))
    return score


def drift(model, ref):
    return float(sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), ref.parameters())))


def main():
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    ref = v1b()
    rows, base = [], {}
    for w, seeds in WORLDS.items():
        te = d[f"test/{w}"]
        allf = {"all": np.ones(len(te["X"]), bool)}
        base[w] = m18b.metrics(ref, te, te["X"], inp, et_mid, allf)["all"]
        print(f"{w} pre-trained: S&P {base[w]['sp_next_dry_err_days']:.3f} d (bias {base[w]['sp_log_bias']:+.3f}), "
              f"forecast {base[w]['fc_next_dry_err_days']:.3f} d", flush=True)
        for n in NS:
            for r_, seed in enumerate(seeds):
                lg = m18b.logger_set(w, inp, seed + 10 * n, n)
                for name, kw in RECIPES.items():
                    t = time.time()
                    if name == "early-stop":
                        perm = np.random.default_rng(seed).permutation(len(lg["theta"]))
                        k = int(round(0.75 * len(perm)))
                        fit, hold = subset(lg, perm[:k]), subset(lg, perm[k:])
                        ft = kp.fine_tune_v1(v1b(), fit, d["train"], seed=r_, val_fn=sp_score_fn(hold, inp))
                    else:
                        ft = kp.fine_tune_v1(v1b(), lg, d["train"], seed=r_, **kw)
                    o = m18b.metrics(ft, te, te["X"], inp, et_mid, allf)["all"]
                    rows.append({"world": w, "n_loggers": n, "draw": r_, "recipe": name, "drift": drift(ft, ref),
                                 "sp_err": o["sp_next_dry_err_days"], "sp_bias": o["sp_log_bias"],
                                 "fc_err": o["fc_next_dry_err_days"], "fc_cov": o["fc_ci90_coverage"],
                                 "fc_ponded_today": o["fc_ponded_today_bacc"]})
                    print(f"{w} n={n} draw {r_} {name:10s}: S&P {o['sp_next_dry_err_days']:.3f} d "
                          f"(bias {o['sp_log_bias']:+.3f}), forecast {o['fc_next_dry_err_days']:.3f} d, "
                          f"drift {rows[-1]['drift']:.3g} ({time.time() - t:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["world", "n_loggers", "recipe"])[["sp_err", "sp_bias", "fc_err", "fc_cov", "fc_ponded_today",
                                                        "drift"]].agg(["mean", "std"])
    print(agg.round(3).to_string())
    summary = {"pretrained": base, "runs": df.round(4).to_dict(orient="records"),
               "mean": {f"{w}|{n}|{r}": {f"{m}_{s}": float(agg.loc[(w, n, r), (m, s)]) for m, s in agg.columns}
                        for w, n, r in agg.index}}
    (OUT / "regularised_fine_tuning.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    recipes = list(RECIPES)
    panels = [("sp_err", "sp_next_dry_err_days", "Season S&P next-dry error (days)"),
              ("sp_bias", "sp_log_bias", "Season S&P bias (log; 0 = unbiased)"),
              ("fc_err", "fc_next_dry_err_days", "Forecast next-dry error (days)")]
    titles = {"reality": "Random mismatch", "reality_systematic": "Random + systematic"}
    colors = {20: "#9ec5f0", 80: "#2a78d6"}
    x = np.arange(len(recipes))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, w in enumerate(WORLDS):
        for ax, (m, bm, lab) in zip(axes[i], panels):
            for j, n in enumerate(NS):
                mu = [summary["mean"][f"{w}|{n}|{r}"][f"{m}_mean"] for r in recipes]
                sd = [summary["mean"][f"{w}|{n}|{r}"][f"{m}_std"] for r in recipes]
                ax.bar(x + (j - 0.5) * 0.38, mu, width=0.36, yerr=sd, color=colors[n], label=f"{n} logger fields",
                       error_kw={"lw": 0.8, "ecolor": "#52514e"})
            ax.axhline(summary["pretrained"][w][bm], color="#52514e", lw=1, ls="--", label="pre-trained, no fine-tuning")
            if m == "sp_bias":
                ax.axhline(0, color="#52514e", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(recipes, fontsize=8)
            ax.set_title(f"{titles[w]}: {lab}", fontsize=9)
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Fine-tuning KGML-Paddy v1b on loggers: five recipes (mean ± sd over 3 logger draws)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figures" / "fig20_regularised_fine_tuning.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "regularised_fine_tuning.json").read_text()))
    else:
        main()
