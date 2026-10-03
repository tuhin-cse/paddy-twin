"""Experiment 19: how many logger fields do the calibrations need?

Three calibrations in KGML-Paddy rest on logger fields (water-level loggers in the campaign):
  1. S&P interval: split-conformal scaling of the season-end effective-S&P interval against each logger
     field's own S&P estimate (kp.logger_sp_eff; Experiment 9);
  2. forecast interval: conformal scaling of the next-dry interval against a logger pseudo-truth (the
     forecast from the logger's level with the S&P law fitted to its recession days; Experiment 12);
  3. N3 prior: logistic recalibration of P(ponded) against the loggers' daily level, before fusing it
     with a satellite pass (Experiment 14).
For each, with n = 3 ... 160 logger fields drawn 30 times from a pool of 300 per world, the calibrated
quantity is evaluated on the Experiment 11 test fields: coverage (target 0.90) and its spread over
draws, interval width, and for N3 the Sentinel-2 wet/dry balanced accuracy under the canopy. Model:
KGML-Paddy v1b as pre-trained (no fine-tuning, so the logger fields are not training data).

Outputs: results/logger_count_calibration.json, results/figures/fig18_logger_count_calibration.png.
Runtime about 5 minutes (`--plot-only` redraws the figure).
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
m14 = __import__("14_n3_realtime")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = {"reality": 52, "reality_systematic": 53}  # logger-pool seeds
POOL = 300
NS = (3, 5, 10, 20, 40, 80, 160)
DRAWS = 30
DAYS = m11.FORECAST_DAYS
K = 100
HW_FLOOR = 0.5
CH = kp.CH


def conformal_q(scores, target=0.9):
    n = len(scores)
    if n < 2:
        return np.nan
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * target) / n)))


def pool_set(w, inp):
    lg = ps.simulate(ps.WORLDS[w], POOL, WORLDS[w], inp)
    lg["H_log"] = ps.end_of_day_depth(lg["logger"])
    lg.update(zip(("h", "tau", "r"), ps.recession_rates(lg["logger"], lg["tp_abs"], inp)))
    return lg


def sp_samples(model, d, inp):
    th, _ = kp.predict_v1(model, d["X"], d["static"], d["X"].shape[1], K)
    return ps.sp_effective(*(th[..., i] for i in range(4)), d["tp_abs"][None, :], inp, dt=0.05)


def forecast_parts(model, d, inp, day, level, theta_for_truth, true_extra=None):
    """Next-dry samples for fields ponded (> 10 mm) at the start of `day`, and their (pseudo-)truth."""
    th, st = kp.predict_v1(model, d["X"], d["static"], day, K)
    h_now = level[:, day - 1]
    sel = np.flatnonzero(np.isfinite(h_now) & (h_now > 10) & (d["X"][:, day - 1, CH["season"]] > 0)
                         & np.isfinite(theta_for_truth[:, 0]))
    h_s = kp.sample_depth(st, day - 1, K, np.random.default_rng(day), fields=sel)
    t0 = inp["s0"] + day
    D = kp.forecast_dry(th[:, sel], h_s, t0, d["tp_abs"][sel][None, :], inp)
    kw = {} if true_extra is None else {"et_f": d["et_f"][sel], "g": d["g"][sel]}
    truth = kp.forecast_dry(theta_for_truth[sel], h_now[sel], t0, d["tp_abs"][sel], inp, **kw)
    return D, truth, sel


def main():
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4)
    inp = ps.load_inputs()
    d = m11.sets(inp)
    model = kp.PaddyNet()
    model.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1b_pretrained.pt"))
    rows = []
    for w in WORLDS:
        t = time.time()
        te, lg = d[f"test/{w}"], pool_set(w, inp)
        T = te["X"].shape[1]
        # 1. S&P interval: per-logger scores, test-field medians and half-widths
        sp_lg, sp_te = sp_samples(model, lg, inp), sp_samples(model, te, inp)
        est = kp.logger_sp_eff(lg, inp)
        med_lg = np.median(sp_lg, 0)
        hw_lg = (np.log(np.quantile(sp_lg, 0.95, 0)) - np.log(np.quantile(sp_lg, 0.05, 0))) / 2
        sp_score = np.where(np.isfinite(est) & (est > 0), np.abs(np.log(np.clip(est, 1e-3, None)) - np.log(med_lg)) / hw_lg, np.nan)
        med_te = np.median(sp_te, 0)
        hw_te = (np.log(np.quantile(sp_te, 0.95, 0)) - np.log(np.quantile(sp_te, 0.05, 0))) / 2
        sp_err_te = np.abs(np.log(te["sp_eff"]) - np.log(med_te)) / hw_te  # covered iff <= q
        # 2. forecast interval: per logger field-day scores; test-field normalised errors
        th_log = kp.logger_theta(lg)
        fc_scores = [[] for _ in range(POOL)]
        fc_err_te = []
        for day in DAYS:
            D, pseudo, sel = forecast_parts(model, lg, inp, day, lg["H_log"], th_log)
            med, hw = np.median(D, 0), np.maximum((np.quantile(D, 0.95, 0) - np.quantile(D, 0.05, 0)) / 2, HW_FLOOR)
            for f, s in zip(sel, np.abs(pseudo - med) / hw):
                fc_scores[f].append(s)
            Dt, truth, _ = forecast_parts(model, te, inp, day, te["H"], te["theta"], true_extra=True)
            medt = np.median(Dt, 0)
            hwt = np.maximum((np.quantile(Dt, 0.95, 0) - np.quantile(Dt, 0.05, 0)) / 2, HW_FLOOR)
            fc_err_te.append(np.maximum(np.abs(truth - medt) - 0.05, 0) / hwt)
        fc_err_te = np.concatenate(fc_err_te)
        # 3. N3 prior recalibration: per logger field (logit prior, ponded) pairs; test passes
        lg_paths = m14.p_wet_paths(model, lg, T)
        pairs = []
        for i in range(POOL):
            xs, ys = [], []
            for day in range(m14.FIRST_DAY, T):
                hl = lg["H_log"][i, day - 1]
                if np.isfinite(hl) and lg["X"][i, day - 1, CH["season"]] > 0:
                    p = np.clip(lg_paths[day][i, day - 1], 1e-4, 1 - 1e-4)
                    xs.append(np.log(p / (1 - p)))
                    ys.append(hl > 0)
            pairs.append((np.array(xs), np.array(ys, float)))
        te_paths = m14.p_wet_paths(model, te, T)
        cfg = ps.WORLDS[w]
        passes = {"prior": [], "wet": [], "truth": [], "rec": [], "spec": [], "lai": []}
        for day in range(m14.FIRST_DAY, T):
            wet_o, dry_o = te["X"][:, day, CH["s2_wet"]] > 0, te["X"][:, day, CH["s2_dry"]] > 0
            f = np.flatnonzero(wet_o | dry_o)
            lai = te["X"][f, day, CH["lai_obs"]] * 5
            rec, spec = kp.pass_rates(inp, "s2", lai, cfg.s2_rec_shift, cfg.s2_spec_shift)
            for k, v in (("prior", te_paths[day][f, day - 1]), ("wet", wet_o[f]),
                         ("truth", te["h6"][f, day * ps.PER_DAY + 2] > 0), ("rec", rec), ("spec", spec), ("lai", lai)):
                passes[k].append(v)
        passes = {k: np.concatenate(v) for k, v in passes.items()}
        canopy = passes["lai"] >= 1.5
        raw_bacc = m11.bacc(passes["truth"][canopy], passes["wet"][canopy])
        print(f"{w}: pool ready ({time.time() - t:.0f} s); raw S2 bacc under canopy {raw_bacc:.3f}", flush=True)

        for n in NS:
            for r in range(DRAWS):
                idx = np.random.default_rng(1000 * n + r).choice(POOL, n, replace=False)
                q_sp = conformal_q(sp_score[idx][np.isfinite(sp_score[idx])])
                q_fc = conformal_q(np.concatenate([np.array(fc_scores[i]) for i in idx]) if n else np.array([]))
                x = np.concatenate([pairs[i][0] for i in idx])
                y = np.concatenate([pairs[i][1] for i in idx])
                a, b = 0.0, 1.0
                for _ in range(30):
                    z = 1 / (1 + np.exp(-(a + b * x)))
                    g = np.array([np.sum(z - y), np.sum((z - y) * x)])
                    wz = z * (1 - z)
                    Hs = np.array([[wz.sum(), (wz * x).sum()], [(wz * x).sum(), (wz * x * x).sum()]]) + 1e-3 * np.eye(2)
                    a, b = np.array([a, b]) - np.linalg.solve(Hs, g)
                lp = np.log(np.clip(passes["prior"], 1e-4, 1 - 1e-4) / np.clip(1 - passes["prior"], 1e-4, 1))
                fused = kp.fuse_pass(1 / (1 + np.exp(-(a + b * lp))), passes["wet"], passes["rec"], passes["spec"])
                rows.append({"world": w, "n_loggers": n, "draw": r,
                             "sp_q": q_sp, "sp_coverage": float(np.mean(sp_err_te <= q_sp)),
                             "fc_q": q_fc, "fc_coverage": float(np.mean(fc_err_te <= q_fc)),
                             "n3_bacc_canopy": m11.bacc(passes["truth"][canopy], fused[canopy] > 0.5),
                             "n3_raw_bacc_canopy": raw_bacc})
        g = pd.DataFrame([r for r in rows if r["world"] == w]).groupby("n_loggers")
        print(g[["sp_coverage", "fc_coverage", "n3_bacc_canopy"]].agg(["mean", lambda s: s.quantile(0.1),
                                                                        lambda s: s.quantile(0.9)]).round(3).to_string(),
              flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["world", "n_loggers"]).agg(
        sp_cov=("sp_coverage", "mean"), sp_cov_p10=("sp_coverage", lambda s: s.quantile(0.1)),
        sp_cov_p90=("sp_coverage", lambda s: s.quantile(0.9)), sp_q=("sp_q", "median"),
        fc_cov=("fc_coverage", "mean"), fc_cov_p10=("fc_coverage", lambda s: s.quantile(0.1)),
        fc_cov_p90=("fc_coverage", lambda s: s.quantile(0.9)), fc_q=("fc_q", "median"),
        n3=("n3_bacc_canopy", "mean"), n3_p10=("n3_bacc_canopy", lambda s: s.quantile(0.1)),
        n3_p90=("n3_bacc_canopy", lambda s: s.quantile(0.9)), n3_raw=("n3_raw_bacc_canopy", "mean")).reset_index()
    summary = {f"{r.world}|{r.n_loggers}": {k: round(float(v), 4) for k, v in r._asdict().items() if k not in ("world", "n_loggers")}
               for r in agg.itertuples(index=False)}
    (OUT / "logger_count_calibration.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    panels = [("sp", "S&P interval coverage\n(target 0.90)"), ("fc", "Forecast interval coverage\n(target 0.90)"),
              ("n3", "N3: S2 wet/dry under canopy\n(balanced acc., LAI ≥ 1.5)")]
    colors = {"reality": "#eb6834", "reality_systematic": "#4a3aa7"}
    labels = {"reality": "Random mismatch", "reality_systematic": "Random + systematic"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, lab) in zip(axes, panels):
        for w in WORLDS:
            ns = [n for n in NS if f"{w}|{n}" in summary]
            mid = [summary[f"{w}|{n}"][key if key == "n3" else f"{key}_cov"] for n in ns]
            lo = [summary[f"{w}|{n}"][f"{key}_p10" if key == "n3" else f"{key}_cov_p10"] for n in ns]
            hi = [summary[f"{w}|{n}"][f"{key}_p90" if key == "n3" else f"{key}_cov_p90"] for n in ns]
            ax.plot(ns, mid, marker="o", ms=4, lw=2, color=colors[w], label=labels[w])
            ax.fill_between(ns, lo, hi, color=colors[w], alpha=0.15, lw=0)
            if key == "n3":
                ax.axhline(summary[f"{w}|{ns[0]}"]["n3_raw"], color=colors[w], lw=1, ls=":")
        if key != "n3":
            ax.axhline(0.9, color="#52514e", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels(NS)
        ax.minorticks_off()
        ax.set_xlabel("Logger fields")
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("How many logger fields the calibrations need (mean and 10-90 % range over 30 draws; "
                 "dotted: raw S2 pass)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figures" / "fig18_logger_count_calibration.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "logger_count_calibration.json").read_text()))
    else:
        main()
