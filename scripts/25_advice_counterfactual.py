"""Experiment 25: advice against practice, in closed loop.

Each simulated village is re-run from transplanting under several irrigation policies (src/awdrt/counterfactual.py):
the farmers' own practice (simulator v2); farmers following KGML-Paddy v1b's advice, with the model updating every
evening from what that arm's fields have shown; the same advice with one regional S&P law, or with no S&P
(ET only); and advice from perfect knowledge. Everything not caused by the policy is shared by every arm.

Advice rules (all followers; 70 % of farmers use the app, as Paddy Twin's default):
  A  "irrigate by tomorrow"     fix the irrigation when the water is forecast gone within 1 day; irrigate that
                                morning (wait 0)
  B  A + asking                 as A, and ask "how much water is in the field now?" where the answer could change
                                the advice (Experiment 13)
  C  B, 3 days ahead            the irrigation is fixed when the water is forecast gone within 3 days (the farmer
                                books the pump), and not revised
  D  B, one dry day             irrigate one day after the water is gone (an AWD-like rule)

Outcomes per field (mean over fields): irrigation water after puddling (mm), irrigations, days with the end-of-day
level 15 cm or more below the surface (stress days, a yield-risk proxy), days ponded, questions asked.
Worlds: the training world and random mismatch; 3 villages of 120 fields each.

Outputs: results/advice_counterfactual.json, results/figures/fig24_advice_counterfactual.png.
Runtime about 10 minutes (`--plot-only` redraws the figure).
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
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import counterfactual as cf  # noqa: E402
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results" / "advice_counterfactual.json"
FIG = ROOT / "results" / "figures" / "fig24_advice_counterfactual.png"
N, SEEDS, SHARE_LOGGING = 120, (101, 102, 103), 0.7
RULES = {"A": dict(wait=0, lead=1.0, ask=False), "B": dict(wait=0, lead=1.0, ask=True),
         "C": dict(wait=0, lead=3.0, ask=True), "D": dict(wait=1, lead=1.0, ask=True)}
RULE_NAMES = {"A": "irrigate by tomorrow", "B": "+ asking", "C": "+ asking, 3 days ahead", "D": "+ asking, one dry day"}
ARMS = ("model", "regional", "et_only", "perfect")
WORLDS = ("prior", "reality")


def run():
    torch.set_num_threads(4)
    m = kp.PaddyNet()
    m.load_state_dict(torch.load(ROOT / "results" / "models" / "kgml_paddy_v1b_pretrained.pt"))
    m.eval()
    inp = ps.load_inputs()
    rows = []
    t0 = time.time()
    for world in WORLDS:
        cfg = replace(ps.WORLDS[world], nisar=False)
        for seed in SEEDS:
            sim = ps.simulate(cfg, N, seed, inp)
            no_log = np.random.default_rng(seed + 1).random(N) >= SHARE_LOGGING
            draws = cf._shared_draws(sim, cfg, inp, seed)
            ftype = sim["ftype"]

            def record(rule, arm, A):
                o = cf.outcomes(A)
                row = {"world": world, "seed": seed, "rule": rule, "arm": arm,
                       **{k: float(o[k].mean()) for k in ("water_mm", "irrigations", "stress_days", "ponded_days", "dry_days")},
                       "asked": float(A["asked"].sum(1).mean()) if "asked" in A else 0.0}
                for k, name in enumerate(("conventional", "awd", "topup")):
                    sel = ftype == k
                    row[f"water_{name}"] = float(o["water_mm"][sel].mean()) if sel.any() else None
                    row[f"stress_{name}"] = float(o["stress_days"][sel].mean()) if sel.any() else None
                rows.append(row)

            with torch.no_grad():
                record("-", "practice", cf.run_arm(sim, cfg, inp, "practice", seed, draws=draws, no_log=no_log))
                for rule, kw in RULES.items():
                    for arm in ARMS:
                        A = cf.run_arm(sim, cfg, inp, arm, seed, kw["wait"], 1.0, m, 24, draws, 0.5, kw["lead"], no_log,
                                       kw["ask"] and arm != "perfect")
                        record(rule, arm, A)
            print(f"{world} seed {seed}: {time.time() - t0:.0f} s", flush=True)
    OUT.write_text(json.dumps({"n_fields": N, "seeds": SEEDS, "share_logging": SHARE_LOGGING, "rules": RULES,
                               "stress_mm": cf.STRESS_MM, "rows": rows}, indent=1))
    return rows


def summary(rows):
    out = {}
    for world in WORLDS:
        base = [r for r in rows if r["world"] == world and r["arm"] == "practice"]
        out[world] = {"practice": {k: float(np.mean([r[k] for r in base])) for k in ("water_mm", "stress_days", "ponded_days")}}
        for rule in RULES:
            for arm in ARMS:
                rr = [r for r in rows if r["world"] == world and r["rule"] == rule and r["arm"] == arm]
                out[world][f"{rule}/{arm}"] = {k: float(np.mean([r[k] for r in rr])) for k in ("water_mm", "stress_days", "ponded_days", "asked")}
    return out


def plot(rows):
    S = summary(rows)
    colors = {"model": "#2a78d6", "regional": "#eb6834", "et_only": "#1baf7a", "perfect": "#52514e"}
    labels = {"model": "KGML-Paddy", "regional": "regional S&P", "et_only": "ET only", "perfect": "perfect knowledge"}
    marks = {"A": "o", "B": "s", "C": "^", "D": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, world in zip(axes, WORLDS):
        p = S[world]["practice"]
        ax.scatter([p["water_mm"]], [p["stress_days"]], s=120, marker="*", color="black", zorder=5)
        ax.annotate("farmers' practice", (p["water_mm"], p["stress_days"]), textcoords="offset points", xytext=(8, 4), fontsize=9)
        for rule in RULES:
            for arm in ARMS:
                v = S[world][f"{rule}/{arm}"]
                ax.scatter([v["water_mm"]], [v["stress_days"]], s=48, marker=marks[rule], color=colors[arm], alpha=0.9,
                           edgecolor="white", linewidth=0.8)
        ax.set_title({"prior": "Training world", "reality": "Random mismatch"}[world])
        ax.set_xlabel("Irrigation water per field (mm, after puddling)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Stress days per field (level below -15 cm)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=colors[a], label=labels[a]) for a in ARMS]
    h += [Line2D([], [], marker=marks[r], ls="", color="#807e78", label=f"{r}: {RULE_NAMES[r]}") for r in RULES]
    h += [Line2D([], [], marker="*", ls="", color="black", markersize=10, label="farmers' practice")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Experiment 25: advice against practice (closed loop, 3 villages x 120 fields, means)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def main():
    rows = json.loads(OUT.read_text())["rows"] if "--plot-only" in sys.argv else run()
    S = summary(rows)
    for world, d in S.items():
        print(world)
        for k, v in d.items():
            print(f"  {k:18s} " + "  ".join(f"{a} {b:7.1f}" for a, b in v.items()))
    plot(rows)


if __name__ == "__main__":
    main()
