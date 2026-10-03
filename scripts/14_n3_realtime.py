"""Experiment 14: N3 in real time. Does the water balance improve satellite wet/dry detection?

N3 claims that requiring satellite detections to be consistent with the water balance improves
wet/dry detection under the canopy. Experiment 11 compared raw passes with a state estimated from
the whole season (past and future). Here everything is causal: at each Sentinel-2 or Sentinel-1
pass on day d (d >= 30, the earliest cut used in training), the water state at the pass time is
estimated with data available by then only:

  raw        the pass's own wet/dry detection;
  prior      KGML-Paddy v1 run on data before day d: P(ponded) at the end of day d-1 (a forecast
             12-18 h ahead, without the pass);
  fused      the prior combined with the pass by Bayes' rule, using the pass's nominal recall and
             specificity at the observed LAI (kp.pass_rates, kp.fuse_pass). This is the N3 check in
             its simplest form: a confident physical prior outweighs an implausible detection. The
             nominal rates are used in every world, including those where real detection is worse;
  fused_cal  calibrated N3, as the campaign could do it: the prior recalibrated on 80 logger fields
             (logistic recalibration of P(ponded) against the loggers' daily levels; the prior is
             overconfident when reality differs from the simulator) and the pass rates of that world
             (what comparing passes with loggers would measure);
  network    KGML-Paddy v1 re-run with data through day d (the pass is one of its inputs), state
             interpolated to the pass time.

Truth is the simulated level at the pass time (12:00 for S2, 18:00 for S1). Balanced accuracy by
sensor, LAI band and world (Experiment 11 test sets, pre-trained v1). Also the quality-control use:
how reliably a confident prior (P < 0.1 or > 0.9) that contradicts a pass flags a wrong detection.

Outputs: results/n3_realtime.json, results/figures/fig13_n3_realtime.png. Runtime about 5 minutes.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m11 = __import__("11_kgml_paddy_v1_state")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = ("prior", "reality", "reality_systematic")
BANDS = ((0, 1.5), (1.5, 3), (3, 9))
SENSORS = {"s2": 2, "s1": 3}  # reading index within the day: 12:00, 18:00
FIRST_DAY = 30
N_LOGGER_FIELDS = 80
LOGGER_SEEDS = {"prior": 21, "reality": 22, "reality_systematic": 23}
CH = kp.CH
ESTIMATORS = ("raw", "prior", "fused", "fused_cal", "network")


def p_wet_paths(model, te, T):
    """P(ponded) path (n, T) for every cut c = FIRST_DAY .. T (observations before day c only)."""
    out = {}
    for c in range(FIRST_DAY, T + 1):
        _, st = kp.predict_v1(model, te["X"], te["static"], c, n_samples=1)
        out[c] = st["p_wet"]
    return out


def platt_fit(model, w, inp, T):
    """Logistic recalibration of the causal prior P(ponded at the end of day d-1 | data before d),
    fitted on N_LOGGER_FIELDS logger fields of world w against their end-of-day logger level."""
    lg = ps.simulate(ps.WORLDS[w], N_LOGGER_FIELDS, LOGGER_SEEDS[w], inp)
    H_log = ps.end_of_day_depth(lg["logger"])
    paths = p_wet_paths(model, lg, T)
    x, y = [], []
    for day in range(FIRST_DAY, T):
        ok = np.isfinite(H_log[:, day - 1]) & (lg["X"][:, day - 1, CH["season"]] > 0)
        p = np.clip(paths[day][ok, day - 1], 1e-4, 1 - 1e-4)
        x.append(np.log(p / (1 - p)))
        y.append(H_log[ok, day - 1] > 0)
    x, y = np.concatenate(x), np.concatenate(y).astype(float)
    a, b = 0.0, 1.0
    for _ in range(50):  # Newton steps for logistic regression y ~ sigmoid(a + b x)
        z = 1 / (1 + np.exp(-(a + b * x)))
        g = np.array([np.sum(z - y), np.sum((z - y) * x)])
        wz = z * (1 - z)
        Hs = np.array([[wz.sum(), (wz * x).sum()], [(wz * x).sum(), (wz * x * x).sum()]]) + 1e-6 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(Hs, g)
    return float(a), float(b)


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    d = m11.sets(inp)
    model = kp.PaddyNet()
    model.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1_pretrained.pt"))
    summary = {}
    for w in WORLDS:
        te = d[f"test/{w}"]
        X, h6 = te["X"], te["h6"]
        T = X.shape[1]
        paths = p_wet_paths(model, te, T)
        pa, pb = platt_fit(model, w, inp, T)
        print(f"{w}: prior recalibration a={pa:.2f}, b={pb:.2f} (identity: 0, 1)", flush=True)
        recs = {e: {s: [] for s in SENSORS} for e in ("truth", "lai", *ESTIMATORS)}
        for s, q in SENSORS.items():
            for day in range(FIRST_DAY, T):
                wet_o, dry_o = X[:, day, CH[f"{s}_wet"]] > 0, X[:, day, CH[f"{s}_dry"]] > 0
                f = np.flatnonzero(wet_o | dry_o)
                if not f.size:
                    continue
                truth = h6[f, day * ps.PER_DAY + q] > 0
                lai = X[f, day, CH["lai_obs"]] * 5
                prior = paths[day][f, day - 1]
                after = paths[day + 1][f]
                wgt = 0.5 if s == "s2" else 0.75  # pass time between the end of day d-1 and of day d
                network = (1 - wgt) * after[:, day - 1] + wgt * after[:, day]
                rec, spec = kp.pass_rates(inp, s, lai)
                fused = kp.fuse_pass(prior, wet_o[f], rec, spec)
                cfg = ps.WORLDS[w]
                shifts = ((cfg.s2_rec_shift, cfg.s2_spec_shift) if s == "s2" else (cfg.s1_acc_shift, cfg.s1_acc_shift))
                rec_c, spec_c = kp.pass_rates(inp, s, lai, *shifts)
                lp = np.log(np.clip(prior, 1e-4, 1 - 1e-4) / np.clip(1 - prior, 1e-4, 1))
                prior_cal = 1 / (1 + np.exp(-(pa + pb * lp)))
                fused_cal = kp.fuse_pass(prior_cal, wet_o[f], rec_c, spec_c)
                for e, v in (("truth", truth), ("lai", lai), ("raw", wet_o[f]), ("prior", prior > 0.5),
                             ("fused", fused > 0.5), ("fused_cal", fused_cal > 0.5), ("network", network > 0.5)):
                    recs[e][s].append(v)
                recs.setdefault("prior_p", {k: [] for k in SENSORS})[s].append(prior)
        res = {}
        for s in SENSORS:
            cat = {e: np.concatenate(recs[e][s]) for e in ("truth", "lai", *ESTIMATORS)}
            pp = np.concatenate(recs["prior_p"][s])
            for lo, hi in BANDS + ((0, 9),):
                b = (cat["lai"] >= lo) & (cat["lai"] < hi)
                key = f"{s}|lai{lo}-{hi}"
                res[key] = {e: m11.bacc(cat["truth"][b], cat[e][b]) for e in ESTIMATORS}
                res[key]["n_passes"] = int(b.sum())
                res[key]["share_wet"] = float(cat["truth"][b].mean())
                # quality control: a confident prior that contradicts the pass
                conf = b & ((pp < 0.1) | (pp > 0.9))
                contra = conf & ((pp > 0.5) != cat["raw"])
                wrong = b & (cat["raw"] != cat["truth"])
                res[key]["qc_flagged_share"] = float(contra.sum() / max(b.sum(), 1))
                res[key]["qc_precision"] = float((contra & wrong).sum() / max(contra.sum(), 1))
                res[key]["qc_recall"] = float((contra & wrong).sum() / max(wrong.sum(), 1))
        res["prior_recalibration"] = {"a": pa, "b": pb}
        summary[w] = res
        print(f"\n== {w}")
        for key, r in ((k, v) for k, v in res.items() if k != "prior_recalibration"):
            print(f"  {key:14s} n={r['n_passes']:6d} wet {r['share_wet']:.2f}  " +
                  "  ".join(f"{e} {r[e]:.3f}" for e in ESTIMATORS) +
                  f"   QC: flags {r['qc_flagged_share']:.2f} of passes, precision {r['qc_precision']:.2f}, "
                  f"recall {r['qc_recall']:.2f}", flush=True)
    (OUT / "n3_realtime.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    colors = {"raw": ("#8a8983", "Raw pass"), "prior": ("#4a3aa7", "Model prior, without the pass"),
              "fused": ("#eb6834", "N3: prior + pass, nominal rates"),
              "fused_cal": ("#1baf7a", "N3: prior + pass, logger-calibrated prior and rates"),
              "network": ("#2a78d6", "Network re-run with the pass")}
    titles = {"prior": "Training world", "reality": "Random mismatch", "reality_systematic": "Random + systematic mismatch"}
    names = {"s2": "Sentinel-2", "s1": "Sentinel-1"}
    x = np.arange(len(BANDS))
    fig, axes = plt.subplots(len(WORLDS), 2, figsize=(10, 8.6), sharex=True, sharey=True)
    for r, w in enumerate(WORLDS):
        for c, s in enumerate(SENSORS):
            ax = axes[r, c]
            for k, e in enumerate(ESTIMATORS):
                col, lab = colors[e]
                ys = [summary[w][f"{s}|lai{lo}-{hi}"][e] for lo, hi in BANDS]
                ax.plot(x, ys, marker="o", ms=5, lw=2, color=col, label=lab)
            ax.axhline(0.5, color="#52514e", lw=0.8, ls=":")
            ax.set_ylim(0.45, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"LAI {lo}–{hi if hi < 9 else ''}".rstrip("–") + ("+" if hi == 9 else "") for lo, hi in BANDS])
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if r == 0:
                ax.set_title(names[s], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{titles[w]}\nwet/dry balanced accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("N3 in real time: wet/dry at each satellite pass, using only data available by then", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig13_n3_realtime.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "n3_realtime.json").read_text()))
    else:
        main()
