"""Experiment 12: fine-tune KGML-Paddy v1's state head on logger fields.

Loggers measure the water level of their own field. Fine-tuning on them can help other fields only
by correcting what differs systematically between simulator and reality (how often irrigations go
unlogged, how detections relate to the water, pump ratings). So everything is tested on fields that
have no logger: the Experiment 11 test fields of the two mismatched worlds.

For n = 10, 40, 80 logger fields (2 draws each), in "reality" and "reality_systematic":
  pretrained             v1 as pre-trained (Experiment 11);
  pretrained_calibrated  + split-conformal scaling of the next-dry interval;
  fine_tuned             kp.fine_tune_v1: logger level (state loss) + logger recession rates (S&P
                         law) + simulated replay;
  fine_tuned_calibrated  + conformal scaling, with scores from cross-fitting (two half-sets of
                         loggers, each scored by a model fine-tuned on the other half), since a
                         model scored on its own training loggers would look too good.

Calibration needs a "true" dry date on logger fields, which a real logger does not give directly
(the farmer irrigates). The pseudo-truth is the forecast from the logger's own level with the S&P law
fitted to its own recession days (kp.logger_theta), exactly what can be computed in the campaign.

Metrics on test fields, averaged over forecasts from days 45, 55 and 65: next-dry error, share off
by > 1 day, 90 % coverage, whether the field is ponded today (causal balanced accuracy), ponded-depth
error; and the season-end effective S&P.

Outputs: results/kgml_paddy_v1_finetune.json, results/figures/fig11_kgml_paddy_v1_finetune.png.
Needs the Experiment 11 cache and weights. Runtime about 30 minutes.
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
m9 = __import__("09_kgml_paddy_v0")
m11 = __import__("11_kgml_paddy_v1_state")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
POOL_CACHE = ROOT / "data" / "cache" / "kgml_v1_pools.pkl"
WORLDS = {"reality": 4, "reality_systematic": 6}  # logger pool seeds (as Experiment 9)
N_LOGGERS = {10: 2, 40: 2, 80: 2}
DAYS = m11.FORECAST_DAYS
N_SAMPLES = 100
CH = kp.CH
# minimum interval half-width (days) in the conformal scores: when the model is sure a field is already
# dry, its interval collapses and a normalised score would explode
HW_FLOOR = 0.5


def pools(inp):
    if POOL_CACHE.exists():
        return pd.read_pickle(POOL_CACHE)
    out = {}
    for w, seed in WORLDS.items():
        p = ps.simulate(ps.WORLDS[w], 2_000, seed, inp)
        p["H_log"] = ps.end_of_day_depth(p["logger"])
        p.update(zip(("h", "tau", "r"), ps.recession_rates(p["logger"], p["tp_abs"], inp)))
        out[w] = {k: v for k, v in p.items() if k not in ("logger", "h6")}
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, POOL_CACHE)
    return out


def subset(d, idx):
    n = len(d["theta"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in d.items()}


def forecast_samples(model, d, inp, day, level_true):
    """Next-dry samples (k, n_sel), truth or pseudo-truth (n_sel,), fields used, ponded-today bacc."""
    th, st = kp.predict_v1(model, d["X"], d["static"], day, N_SAMPLES)
    h_now = level_true[:, day - 1]
    known = np.isfinite(h_now) & (d["X"][:, day - 1, CH["season"]] > 0)
    bacc = m11.bacc(h_now[known] > 0, st["p_wet"][known, day - 1] > 0.5)
    sel = np.flatnonzero(known & (h_now > 10))
    h_s = kp.sample_depth(st, day - 1, N_SAMPLES, np.random.default_rng(day), fields=sel)
    D = kp.forecast_dry(th[:, sel], h_s, inp["s0"] + day, d["tp_abs"][sel][None, :], inp)
    return D, sel, bacc, st


def test_metrics(model, te, inp, et_mid, scale=None):
    """Forecast metrics on the test fields (true truth), averaged over DAYS; scale: conformal factor."""
    rows = []
    for day in DAYS:
        D, sel, bacc, st = forecast_samples(model, te, inp, day, te["H"])
        truth = kp.forecast_dry(te["theta"][sel], te["H"][sel, day - 1], inp["s0"] + day, te["tp_abs"][sel], inp,
                                et_f=te["et_f"][sel], g=te["g"][sel])
        med, lo, hi = np.median(D, 0), np.quantile(D, 0.05, 0), np.quantile(D, 0.95, 0)
        if scale is not None:
            hw = np.maximum((hi - lo) / 2, HW_FLOOR)
            lo, hi = med - scale * hw, med + scale * hw
        err = np.abs(med - truth)
        H, wet = te["H"][:, day - 1], st["p_wet"][:, day - 1] > 0.5
        both = np.isfinite(H) & (H > 0) & wet
        rows.append({"next_dry_err_days": float(np.median(err)), "share_err_gt1d": float((err > 1).mean()),
                     "ci90_coverage": float(((lo <= truth + 0.05) & (truth - 0.05 <= hi)).mean()),
                     "ponded_today_bacc": bacc,
                     "ponded_depth_mae_mm": float(np.mean(np.abs(st["pos_mu"][both, day - 1] - H[both])))})
    return pd.DataFrame(rows).mean().to_dict()


def season_sp(model, te, inp, et_mid):
    th, _ = kp.predict_v1(model, te["X"], te["static"], te["X"].shape[1], N_SAMPLES)
    m = m9.evaluate(th, te, inp, et_mid)
    return {"sp_next_dry_err_days": m["next_dry_err_days"], "sp_spearman": m["spearman"]}


def conformal_scores(model, lg, inp):
    """Normalised scores |pseudo-truth - median| / half-width on logger fields, all DAYS."""
    th_log = kp.logger_theta(lg)
    scores = []
    for day in DAYS:
        D, sel, _, _ = forecast_samples(model, lg, inp, day, lg["H_log"])
        ok = np.isfinite(th_log[sel, 0])
        if not ok.any():
            continue
        f = sel[ok]
        pseudo = kp.forecast_dry(th_log[f], lg["H_log"][f, day - 1], inp["s0"] + day, lg["tp_abs"][f], inp)
        med, lo, hi = np.median(D[:, ok], 0), np.quantile(D[:, ok], 0.05, 0), np.quantile(D[:, ok], 0.95, 0)
        scores.append(np.abs(pseudo - med) / np.maximum((hi - lo) / 2, HW_FLOOR))
    return np.concatenate(scores) if scores else np.array([])


def conformal_q(scores, target=0.9):
    n = len(scores)
    if n < 5:
        return 1.0
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * target) / n)))


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    pl = pools(inp)
    base = kp.PaddyNet()
    base.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1_pretrained.pt"))
    state0 = {k: v.clone() for k, v in base.state_dict().items()}

    def fresh():
        m = kp.PaddyNet()
        m.load_state_dict(state0)
        return m

    runs = []
    for w in WORLDS:
        te, pool = d[f"test/{w}"], pl[w]
        t = time.time()
        pre = {**test_metrics(fresh(), te, inp, et_mid), **season_sp(fresh(), te, inp, et_mid)}
        runs.append({"world": w, "model": "pretrained", "n_loggers": 0, "draw": 0, **pre})
        print(f"{w} pretrained: " + ", ".join(f"{k} {v:.3f}" for k, v in pre.items()) + f"  ({time.time() - t:.0f} s)",
              flush=True)
        eligible = np.flatnonzero(np.isfinite(pool["r"]).sum(1) >= 8)
        for n, reps in N_LOGGERS.items():
            for r_ in range(reps):
                t = time.time()
                idx = np.random.default_rng(1000 * n + r_).choice(eligible, n, replace=False)
                lg = subset(pool, idx)
                row = {"world": w, "n_loggers": n, "draw": r_}
                q_pre = conformal_q(conformal_scores(fresh(), lg, inp))
                runs.append({**row, "model": "pretrained_calibrated", "conformal_scale": q_pre,
                             **test_metrics(fresh(), te, inp, et_mid, q_pre)})
                # cross-fitted scores for the fine-tuned model
                half = np.random.default_rng(r_).permutation(n)
                a_idx, b_idx = half[: n // 2], half[n // 2:]
                sc = []
                for fit_i, score_i in ((a_idx, b_idx), (b_idx, a_idx)):
                    mh = kp.fine_tune_v1(fresh(), subset(lg, fit_i), d["train"], seed=r_)
                    sc.append(conformal_scores(mh, subset(lg, score_i), inp))
                q_ft = conformal_q(np.concatenate(sc))
                ft = kp.fine_tune_v1(fresh(), lg, d["train"], seed=r_)
                m_ft = {**test_metrics(ft, te, inp, et_mid), **season_sp(ft, te, inp, et_mid)}
                runs.append({**row, "model": "fine_tuned", **m_ft})
                runs.append({**row, "model": "fine_tuned_calibrated", "conformal_scale": q_ft,
                             **test_metrics(ft, te, inp, et_mid, q_ft)})
                print(f"{w} n={n} draw={r_}: fine-tuned next-dry {m_ft['next_dry_err_days']:.2f} d, ponded-today "
                      f"{m_ft['ponded_today_bacc']:.3f}, depth MAE {m_ft['ponded_depth_mae_mm']:.1f} mm, "
                      f"cov {m_ft['ci90_coverage']:.2f} -> {runs[-1]['ci90_coverage']:.2f} (q {q_ft:.2f}); "
                      f"pre-trained cal cov {runs[-3]['ci90_coverage']:.2f} (q {q_pre:.2f})  ({time.time() - t:.0f} s)",
                      flush=True)

    df = pd.DataFrame(runs)
    agg = df.groupby(["world", "model", "n_loggers"]).mean(numeric_only=True).drop(columns="draw")
    summary = {f"{w}|{m}|{n}": {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
               for (w, m, n), row in agg.iterrows()}
    (OUT / "kgml_paddy_v1_finetune.json").write_text(json.dumps(summary, indent=2))
    print(agg.round(3).to_string())

    plot(summary)


def plot(summary):
    """Figure from the saved summary (python scripts/12_kgml_paddy_v1_finetune.py --plot-only)."""
    get = lambda w, m, n, k: summary[f"{w}|{m}|{n}"][k]  # noqa: E731
    colors = {"pretrained_calibrated": ("#2a78d6", "-"), "fine_tuned": ("#eb6834", "--"),
              "fine_tuned_calibrated": ("#eb6834", "-")}
    labels = {"pretrained_calibrated": "Pre-trained + calibrated", "fine_tuned": "Fine-tuned on loggers (raw intervals)",
              "fine_tuned_calibrated": "Fine-tuned + calibrated"}
    titles = {"reality": "Random mismatch", "reality_systematic": "Random + systematic mismatch"}
    metrics = [("next_dry_err_days", "Next-dry error, median (days)", (0.8, 1.2)),
               ("ponded_depth_mae_mm", "Ponded depth error, MAE (mm)", (8, 16)),
               ("ci90_coverage", "90 % interval coverage", (0.65, 1.02))]
    xs = list(N_LOGGERS)
    fig, axes = plt.subplots(len(WORLDS), 3, figsize=(12, 6.6), sharex=True)
    for r, w in enumerate(WORLDS):
        for c, (metric, ylab, ylim) in enumerate(metrics):
            ax = axes[r, c]
            for mname, (col, ls) in colors.items():
                if metric != "ci90_coverage" and mname != "fine_tuned":
                    continue  # calibration only widens intervals; the point estimate is the fine-tuned one
                ax.plot(xs, [get(w, mname, n, metric) for n in xs], marker="o", ms=4, lw=2, color=col, ls=ls,
                        label=labels[mname])
            ax.axhline(get(w, "pretrained", 0, metric), color="#2a78d6", lw=2, ls=":", label="Pre-trained (no loggers)")
            if metric == "ci90_coverage":
                ax.axhline(0.9, color="#52514e", lw=1, ls="--")
            ax.set_ylim(*ylim)
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(xs)
            ax.minorticks_off()
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.set_ylabel(f"{titles[w]}\n{ylab}" if c == 0 else ylab)
            if r == len(WORLDS) - 1:
                ax.set_xlabel("Logger fields")
    h, l = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("KGML-Paddy v1 fine-tuned on logger fields; tested on fields without loggers", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig11_kgml_paddy_v1_finetune.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "kgml_paddy_v1_finetune.json").read_text()))
    else:
        main()
