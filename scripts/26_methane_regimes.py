"""Experiment 26: field water regimes and methane (IPCC Tier 1), and H5b.

For simulated villages (3 x 120 fields, training world and random mismatch), each field's water regime during
the season (continuously flooded, single or multiple drainage; IPCC 2019 Refinement) is classified from:
  truth       the simulator's end-of-day water level
  model       KGML-Paddy v1b's causal daily belief (ponded when P > 0.5, with the NISAR answer update)
  satellite   the satellite calls alone (Sentinel-2, Sentinel-1, NISAR), each day taking the last call
and compared with one regional default for every field: the IPCC aggregated "irrigated" factor (regime unknown,
0.60) or continuously flooded. Tier 1 methane (src/awdrt/methane.py) follows from the regime.

Measures: regime agreement with the truth (balanced over the true regimes present); recall of continuously
flooded fields (the highest emitters); the village total's error; and, for H5b, the share of fields whose
estimate is off by at least half the gap between continuous flooding and AWD. Sensitivity: a drainage period of
at least 1, 2 (default) or 3 days without standing water. And the closed loop (Experiment 25, rule B): how the
advice changes methane, against the farmers' practice and perfect knowledge.

Outputs: results/methane_regimes.json, results/figures/fig25_methane_regimes.png. Runtime about 5 minutes.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app import engine  # noqa: E402

OUT = ROOT / "results" / "methane_regimes.json"
FIG = ROOT / "results" / "figures" / "fig25_methane_regimes.png"
SEEDS, WORLDS, MIN_DRY = (201, 202, 203), ("prior", "reality"), (1, 2, 3)


def run():
    torch.set_num_threads(4)
    inp = engine.ps.load_inputs()
    rows, loop = [], []
    for world in WORLDS:
        for seed in SEEDS:
            R = engine.run_season({"n_fields": 120, "world": world, "seed": seed}, inp)
            for md in MIN_DRY:
                for regional in ("aggregated", "continuous"):
                    M = engine.run_methane(R["run"], {"min_dry": md, "regional": regional})
                    S = M["sources"]
                    row = {"world": world, "seed": seed, "min_dry": md, "default": regional,
                           "truth_counts": S["truth"]["counts"], "truth_kg": S["truth"]["total_kg"]}
                    for k in ("model", "satellite", "regional"):
                        v = S[k]["vs_truth"]
                        row[k] = {"counts": S[k]["counts"], "total_err": v["total_err"], "balanced": v["regime_balanced"],
                                  "cf_recall": v["cf_recall"], "half_gap_share": v["half_gap_share"], "field_err": v["field_err_kg_ha"]}
                    rows.append(row)
            C = engine.run_counterfactual(R["run"], {"ask": True})
            loop.append({"world": world, "seed": seed, **{a: C["arms"][a]["methane"] | {"water_mm": C["arms"][a]["water_mm"],
                                                                                          "stress_days": C["arms"][a]["stress_days"]}
                                                           for a in ("practice", "model", "perfect")}})
            for a in ("practice", "model", "perfect"):
                loop[-1][a].pop("field_kg_ha", None)
            print(f"{world} seed {seed} done", flush=True)
    OUT.write_text(json.dumps({"seeds": SEEDS, "rows": rows, "closed_loop": loop}, indent=1))
    return rows, loop


def mean(rows, f):
    return float(np.mean([f(r) for r in rows]))


def report(rows, loop):
    for world in WORLDS:
        print(world)
        for md in MIN_DRY:
            rr = [r for r in rows if r["world"] == world and r["min_dry"] == md and r["default"] == "aggregated"]
            tc = {g: mean(rr, lambda r: r["truth_counts"][g]) for g in ("continuous", "single", "multiple")}
            print(f"  min_dry {md}: true regimes {tc}")
            for k in ("model", "satellite", "regional"):
                print(f"    {k:9s} balanced {mean(rr, lambda r: r[k]['balanced'] or 0):.2f}  cf_recall "
                      f"{mean(rr, lambda r: r[k]['cf_recall'] or 0):.2f}  total_err {mean(rr, lambda r: r[k]['total_err']):+.3f}  "
                      f"half_gap {mean(rr, lambda r: r[k]['half_gap_share']):.2f}  field_err {mean(rr, lambda r: r[k]['field_err']):.1f}")
            rc = [r for r in rows if r["world"] == world and r["min_dry"] == md and r["default"] == "continuous"]
            print(f"    regional=continuous total_err {mean(rc, lambda r: r['regional']['total_err']):+.3f} "
                  f"half_gap {mean(rc, lambda r: r['regional']['half_gap_share']):.2f}")
        ll = [x for x in loop if x["world"] == world]
        for a in ("practice", "model", "perfect"):
            print(f"  closed loop {a:9s} kg/ha {mean(ll, lambda x: x[a]['kg_ha']):.1f}  CF fields {mean(ll, lambda x: x[a]['counts']['continuous']):.1f}"
                  f"  water {mean(ll, lambda x: x[a]['water_mm']):.0f}  stress {mean(ll, lambda x: x[a]['stress_days']):.1f}")


def plot(rows, loop):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    srcs = [("model", "KGML-Paddy", "#2a78d6"), ("satellite", "satellite only", "#eb6834"), ("regional", "regional default", "#807e78")]
    x = np.arange(len(WORLDS))
    for j, (k, lab, col) in enumerate(srcs):
        rr = lambda w: [r for r in rows if r["world"] == w and r["min_dry"] == 2 and r["default"] == "aggregated"]  # noqa: E731
        if k != "regional":
            axes[0].bar(x + (j - 0.5) * 0.35, [mean(rr(w), lambda r: r[k]["balanced"]) for w in WORLDS], 0.33, color=col, label=lab)
        axes[1].bar(x + (j - 1) * 0.27, [mean(rr(w), lambda r: r[k]["half_gap_share"]) for w in WORLDS], 0.25, color=col, label=lab)
    axes[0].set_title("Regime right (balanced), drainage >= 2 days")
    axes[1].set_title("Fields off by >= half the CF-AWD gap (H5b)")
    for ax in axes[:2]:
        ax.set_xticks(x, ["training world", "random mismatch"])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25)
    for w, mk in zip(WORLDS, ("o", "s")):
        ll = [l_ for l_ in loop if l_["world"] == w]
        for a, col in (("practice", "black"), ("model", "#2a78d6"), ("perfect", "#52514e")):
            axes[2].scatter([mean(ll, lambda l_: l_[a]["water_mm"])], [mean(ll, lambda l_: l_[a]["kg_ha"])], marker=mk, color=col, s=50,
                            label=f"{a} ({'training' if w == 'prior' else 'mismatch'})")
    axes[2].set_xlabel("irrigation water per field (mm)")
    axes[2].set_ylabel("methane, kg CH4/ha (Tier 1)")
    axes[2].set_title("Closed loop: advice with asking (Exp. 25 rule B)")
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Experiment 26: field water regimes and methane (3 villages x 120 fields per world, means)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def main():
    if "--plot-only" in sys.argv:
        d = json.loads(OUT.read_text())
        rows, loop = d["rows"], d["closed_loop"]
    else:
        rows, loop = run()
    report(rows, loop)
    plot(rows, loop)


if __name__ == "__main__":
    main()
