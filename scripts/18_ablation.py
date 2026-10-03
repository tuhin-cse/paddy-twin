"""Experiment 18: full ablation of KGML-Paddy for paper 2.

The evaluation design, after the component attribution of KGML-ag-Carbon (Liu et al.
2024, Fig. 3): switch each component of the full model (v1b, Experiment 17) on and off.
  S  satellite input (wet/dry passes and satellite LAI; off: removed from inputs);
  P  pre-training on 20,000 simulated fields (off: trained from scratch on 40 logger fields only:
     logger levels, plus the S&P target from the logger's recession days);
  F  physical structure: the S&P law SP(h, tau) = a(1 + w tau/80)e^(-c tau) + b h, integrated by the
     forecast (off: a black-box head predicts the effective S&P, used as a constant loss rate);
  L  the physics-consistency loss (off: weight 0).
With pre-training: all 2x2x2 combinations of S, F, L (v1b is S1-F1-L1). Without: the four S x F
combinations (with L), trained on 40 logger fields per world.

Evaluated on the Experiment 11 test sets (all three worlds):
  season-end effective S&P: next-dry error, rank correlation;
  forecasts from days 45, 55, 65 (data before that day): next-dry error for all in-season fields,
  ponded today, the irrigate-by-tomorrow call, 90 % coverage.
Main effects: mean change when a component is switched on, averaged over the other components (with
pre-training), and the effect of pre-training itself (P0 vs P1 at L on).

Outputs: results/ablation.json, results/figures/fig17_ablation.png; weights in results/models/ablation/.
Runtime about 60 minutes (6 pre-trainings of 6-9 minutes; cached afterwards). `--plot-only` redraws the figure.
Follow-up (non-logging fields, fine-tuning): scripts/18b_ablation_followup.py.
"""

import itertools
import json
import pathlib
import shutil
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
m17 = __import__("17_train_with_nonloggers")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WDIR = OUT / "models" / "ablation"
WORLDS = ("prior", "reality", "reality_systematic")
LOGGER_SEEDS = {"prior": 41, "reality": 42, "reality_systematic": 43}
N_LOGGERS = 40
K = 100
CH = kp.CH


def name(S, P, F, L):
    return f"S{S}-P{P}-F{F}-L{L}"


def pretrained(S, F, L, d):
    path = WDIR / f"{name(S, 1, F, L)}.pt"
    model = kp.PaddyNet(structure=bool(F))
    if not path.exists():
        if (S, F, L) == (1, 1, 1):  # the full model is v1b
            shutil.copy(OUT / "models" / "kgml_paddy_v1b_pretrained.pt", path)
        else:
            t = time.time()
            m = kp.pretrain_v1(d["train"], d["val"], epochs=30, p_nolog=m17.P_NOLOG, structure=bool(F),
                               sat_off=not S, phys_w=0.01 if L else 0.0, log=lambda _: None)
            torch.save(m.state_dict(), path)
            print(f"  pre-trained {name(S, 1, F, L)} in {time.time() - t:.0f} s", flush=True)
    model.load_state_dict(torch.load(path))
    return model


def logger_set(w, inp):
    lg = ps.simulate(ps.WORLDS[w], 2 * N_LOGGERS, LOGGER_SEEDS[w], inp)
    lg["H_log"] = ps.end_of_day_depth(lg["logger"])
    lg.update(zip(("h", "tau", "r"), ps.recession_rates(lg["logger"], lg["tp_abs"], inp)))
    lg["sp_log"] = kp.logger_sp_eff(lg, inp)
    keep = np.flatnonzero(np.isfinite(lg["r"]).sum(1) >= 8)[:N_LOGGERS]
    n = len(lg["theta"])
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in lg.items()
            if k not in ("logger", "h6")}


def evaluate(model, te, X, inp, et_mid):
    th, _ = kp.predict_v1(model, X, te["static"], X.shape[1], K)
    sp = np.median(ps.sp_effective(*(th[..., i] for i in range(4)), te["tp_abs"][None, :], inp, dt=0.05), 0)
    m = m9.point_metrics(sp, te["sp_eff"], et_mid)
    fc = m17.forecast_group(model, te, X, inp, {"all": np.ones(len(X), bool)})["all"]
    return {"sp_next_dry_err_days": m["next_dry_err_days"], "sp_spearman": float(spearmanr(sp, te["sp_eff"]).correlation),
            "forecast_next_dry_err_days": fc["next_dry_err_days"], "ponded_today_bacc": fc["ponded_today_bacc"],
            "irrigate_by_tomorrow_bacc": fc["irrigate_by_tomorrow_bacc"], "forecast_ci90_coverage": fc["ci90_coverage"]}


def main():
    torch.set_num_threads(4)
    WDIR.mkdir(parents=True, exist_ok=True)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    models = {}
    for S, F, L in itertools.product((0, 1), repeat=3):
        models[name(S, 1, F, L)] = pretrained(S, F, L, d)
    rows = []
    for w in WORLDS:
        te = d[f"test/{w}"]
        lg = logger_set(w, inp)
        variants = dict(models)
        for S, F in itertools.product((0, 1), repeat=2):
            lgS = {**lg, "X": kp.drop_satellite(lg["X"]) if not S else lg["X"]}
            variants[name(S, 0, F, 1)] = kp.train_loggers_only(kp.PaddyNet(structure=bool(F)), lgS, seed=S * 2 + F)
        for vname, model in variants.items():
            t = time.time()
            S = int(vname[1])
            X = kp.drop_satellite(te["X"]) if not S else te["X"]
            r = evaluate(model, te, X, inp, et_mid)
            rows.append({"world": w, "variant": vname, "S": S, "P": int(vname[4]), "F": int(vname[7]), "L": int(vname[10]), **r})
            print(f"{w} {vname}: S&P {r['sp_next_dry_err_days']:.2f} d / rho {r['sp_spearman']:.2f} | forecast "
                  f"{r['forecast_next_dry_err_days']:.2f} d, ponded-today {r['ponded_today_bacc']:.3f}, "
                  f"irrigate-call {r['irrigate_by_tomorrow_bacc']:.3f}, cov {r['forecast_ci90_coverage']:.2f}  "
                  f"({time.time() - t:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    metrics = ["sp_next_dry_err_days", "sp_spearman", "forecast_next_dry_err_days", "ponded_today_bacc",
               "irrigate_by_tomorrow_bacc", "forecast_ci90_coverage"]
    effects = {}
    for w in WORLDS:
        g = df[(df.world == w) & (df.P == 1)]
        eff = {c: {m: float(g[g[c] == 1][m].mean() - g[g[c] == 0][m].mean()) for m in metrics} for c in ("S", "F", "L")}
        gp = df[(df.world == w) & (df.L == 1)]
        eff["P"] = {m: float(gp[gp.P == 1][m].mean() - gp[gp.P == 0][m].mean()) for m in metrics}
        effects[w] = eff
    summary = {"rows": df.round(4).to_dict(orient="records"), "main_effects": effects}
    (OUT / "ablation.json").write_text(json.dumps(summary, indent=2))
    print(df.set_index(["world", "variant"])[metrics].round(3).to_string())
    print(json.dumps({w: {c: {m: round(v, 3) for m, v in e.items()} for c, e in eff.items()} for w, eff in effects.items()},
                     indent=1))
    plot(summary)


def plot(summary):
    eff = summary["main_effects"]
    comps = [("P", "Pre-training\n(vs 40 loggers only)"), ("S", "Satellite input"), ("F", "Physical structure\n(S&P law)"),
             ("L", "Physics-\nconsistency loss")]
    panels = [("sp_next_dry_err_days", "Season S&P: next-dry error (days)\nnegative = better"),
              ("forecast_next_dry_err_days", "Forecast next-dry error (days)\nnegative = better"),
              ("ponded_today_bacc", "Ponded today? (balanced acc.)\npositive = better"),
              ("irrigate_by_tomorrow_bacc", "Irrigate-by-tomorrow call\npositive = better")]
    colors = {"prior": "#2a78d6", "reality": "#eb6834", "reality_systematic": "#4a3aa7"}
    labels = {"prior": "Training world", "reality": "Random mismatch", "reality_systematic": "Random + systematic"}
    y = np.arange(len(comps))
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.2), sharey=True)
    for ax, (m, lab) in zip(axes, panels):
        for j, w in enumerate(WORLDS):
            ax.barh(y + (j - 1) * 0.25, [eff[w][c][m] for c, _ in comps], height=0.24, color=colors[w], label=labels[w])
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([c[1] for c in comps], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("KGML-Paddy ablation: effect of switching each component on (averaged over the other components)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figures" / "fig17_ablation.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "ablation.json").read_text()))
    else:
        main()
