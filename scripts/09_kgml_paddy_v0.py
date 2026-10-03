"""Experiment 9: KGML-Paddy v0 on simulator v2.

Pre-train the PyTorch model (awdrt.kgml_paddy) on 20,000 fields from the "prior" world of
simulator v2 (awdrt.paddy_sim), then test on two mismatched realities:
  reality             the Experiment 8 strong + observation mismatches and a shifted parameter
                      prior; the observation errors are random per field;
  reality_systematic  the same, plus errors shared by every field (every pump's rating 25 % high,
                      more false-wet satellite detections), which loggers could correct.
With n logger fields, compare: split-conformal calibration of the pre-trained posterior's
intervals against each logger field's own effective S&P (no retraining); fine-tuning through the physics (recession-rate loss) with simulated
replay, with and without calibration; the same network trained on the loggers alone; and a
regional constant. This repeats Experiment 8 with the sequence
network planned for the method, and on the richer simulator.

Truth and prediction are the effective S&P of a 50 mm irrigation on 1 March, as in Experiment 8;
predictions come from 100 posterior samples pushed through the physics.

Outputs: results/kgml_paddy_v0.json, results/kgml_paddy_v0_runs.csv,
results/models/kgml_paddy_v0_pretrained.pt, results/figures/fig8_kgml_paddy_v0.png.
Runtime about 40 minutes on 4 CPU cores (pre-training about 5 of them; cached afterwards).
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
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "kgml_v0_sets.pkl"
N_LOGGERS = {5: 3, 10: 3, 20: 3, 40: 3, 80: 2}
N_SAMPLES = 100


WORLD_SEEDS = {"reality": (4, 5), "reality_systematic": (6, 7)}  # (logger pool, test)


def sets(inp):
    out = pd.read_pickle(CACHE) if CACHE.exists() else {}
    changed = False
    if "train" not in out:
        out.update(train=ps.simulate(ps.WORLDS["prior"], 20_000, 1, inp), val=ps.simulate(ps.WORLDS["prior"], 2_000, 2, inp),
                   prior_test=ps.simulate(ps.WORLDS["prior"], 2_000, 3, inp))
        changed = True
    for wname, (sp, st) in WORLD_SEEDS.items():
        if f"{wname}/pool" not in out:
            pool = ps.simulate(ps.WORLDS[wname], 2_000, sp, inp)
            pool.update(zip(("h", "tau", "r"), ps.recession_rates(pool["logger"], pool["tp_abs"], inp)))
            out[f"{wname}/pool"], out[f"{wname}/test"] = pool, ps.simulate(ps.WORLDS[wname], 2_000, st, inp)
            changed = True
    for k in list(out):  # loggers only matter for the pools
        if not k.endswith("pool"):
            out[k] = {kk: v for kk, v in out[k].items() if kk not in ("h6", "logger")}
    if changed:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(out, CACHE)
    return out


def subset(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(d["theta"]),) else v) for k, v in d.items()}


def evaluate(samples, data, inp, et_mid, scale=None):
    """samples (k, n, 4) of theta -> metrics on effective S&P and next-dry error.
    scale: conformal factor for the 90 % log-interval half-width (None = raw posterior)."""
    a, b, c, w = (samples[..., i] for i in range(4))
    sp = ps.sp_effective(a, b, c, w, data["tp_abs"][None, :], inp, dt=0.05)
    med, lo, hi = np.median(sp, 0), np.quantile(sp, 0.05, 0), np.quantile(sp, 0.95, 0)
    if scale is not None:
        hw = (np.log(hi) - np.log(lo)) / 2
        lo, hi = med * np.exp(-scale * hw), med * np.exp(scale * hw)
    return point_metrics(med, data["sp_eff"], et_mid, lo, hi) | {
        f"rho_{p}": float(spearmanr(np.median(samples[..., i], 0), data["theta"][:, i]).correlation)
        for i, p in enumerate("abcw")}


def point_metrics(pred, y, et_mid, lo=None, hi=None):
    kc2 = ps.kc_rice(2.0)
    D = 50 / (kc2 * et_mid + y)
    err = np.abs(50 / (kc2 * et_mid + np.clip(pred, 0, None)) - D)
    m = {"spearman": float(spearmanr(pred, y).correlation) if np.ptp(pred) > 1e-9 else 0.0,
         "median_abs_pct_err": float(np.median(np.abs(pred - y) / y) * 100),
         "next_dry_err_days": float(np.median(err)), "share_err_gt1d": float((err > 1).mean())}
    if lo is not None:
        m["ci90_coverage"] = float(((lo <= y) & (y <= hi)).mean())
    return m


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = sets(inp)

    wpath = OUT / "models" / "kgml_paddy_v0_pretrained.pt"
    if wpath.exists():
        model = kp.Encoder()
        model.load_state_dict(torch.load(wpath))
    else:
        t = time.time()
        model = kp.pretrain(d["train"], d["val"], epochs=30, log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), wpath)
        print(f"pre-trained in {time.time() - t:.0f} s", flush=True)
    state0 = {k: v.clone() for k, v in model.state_dict().items()}

    summary = {"pretrained_in_distribution": evaluate(kp.posterior(model, d["prior_test"]["X"], d["prior_test"]["static"],
                                                                   N_SAMPLES), d["prior_test"], inp, et_mid),
               "world_truth": {}}
    print("pretrained, prior world:", {k: round(v, 3) for k, v in summary["pretrained_in_distribution"].items()}, flush=True)
    runs = []
    for wname in WORLD_SEEDS:
        pool, test = d[f"{wname}/pool"], d[f"{wname}/test"]
        summary["world_truth"][wname] = {"sp_eff_median": float(np.median(test["sp_eff"])),
                                         "sp_eff_p10_p90": np.percentile(test["sp_eff"], [10, 90]).round(2).tolist()}
        ev = lambda mdl, scale=None: evaluate(kp.posterior(mdl, test["X"], test["static"], N_SAMPLES),  # noqa: E731
                                              test, inp, et_mid, scale)
        sp_log_pool = kp.logger_sp_eff(pool, inp)
        model.load_state_dict(state0)
        pre = ev(model)
        print(f"{wname}: pretrained", {k: round(v, 3) for k, v in pre.items()}, flush=True)
        runs += [{"world": wname, "model": "pretrained", "n_loggers": 0, "draw": 0, **pre},
                 {"world": wname, "model": "regional", "n_loggers": 0, "draw": 0,
                  **point_metrics(np.full(len(test["sp_eff"]), 5.0), test["sp_eff"], et_mid)}]
        has_rec = np.isfinite(pool["r"]).sum(1) >= 5  # a logger field needs some clean recession days
        for n, reps in N_LOGGERS.items():
            for r_ in range(reps):
                idx = np.random.default_rng(100 * n + r_).choice(np.flatnonzero(has_rec), n, replace=False)
                lg = subset(pool, idx)
                row = {"world": wname, "n_loggers": n, "draw": r_}
                runs.append({**row, "model": "regional",
                             **point_metrics(np.full(len(test["sp_eff"]), float(np.median(np.nanmean(lg["r"], 1)))),
                                             test["sp_eff"], et_mid)})
                model.load_state_dict(state0)
                # conformal factor from the pre-trained model, which never saw these loggers; reused for the
                # fine-tuned model (calibrating it on its own training loggers would be circular)
                q = kp.conformal_scale(kp.sp_eff_samples(model, lg, inp, N_SAMPLES), sp_log_pool[idx])
                runs.append({**row, "model": "pretrained_calibrated", "conformal_scale": q, **ev(model, q)})
                ft = kp.fine_tune(model, lg, d["train"], steps=300, seed=r_)
                runs.append({**row, "model": "fine_tuned", **ev(ft)})
                runs.append({**row, "model": "fine_tuned_calibrated", "conformal_scale": q, **ev(ft, q)})
                torch.manual_seed(r_)
                scratch = kp.fine_tune(kp.Encoder(), lg, None, steps=600, lr_enc=1e-3, seed=r_)
                runs.append({**row, "model": "loggers_only", **ev(scratch)})
                print(f"{wname} n={n} draw={r_}: " + ", ".join(
                    f"{x['model']} {x['next_dry_err_days']:.2f} d/{x.get('ci90_coverage', float('nan')):.2f}" for x in runs[-5:]),
                    flush=True)

    df = pd.DataFrame(runs)
    df.to_csv(OUT / "kgml_paddy_v0_runs.csv", index=False)
    agg = df.groupby(["world", "model", "n_loggers"]).mean(numeric_only=True).drop(columns="draw")
    summary["by_world_model_loggers"] = {f"{w}|{m}|{n}": {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
                                         for (w, m, n), row in agg.iterrows()}
    (OUT / "kgml_paddy_v0.json").write_text(json.dumps(summary, indent=2))
    print(agg[["spearman", "next_dry_err_days", "share_err_gt1d", "ci90_coverage", "median_abs_pct_err", "conformal_scale"]]
          .round(3).to_string())

    # figure: rows = worlds; columns = next-dry error, rank correlation, interval coverage
    colors = {"pretrained": "#2a78d6", "pretrained_calibrated": "#1baf7a", "fine_tuned": "#eb6834",
              "fine_tuned_calibrated": "#eda100", "loggers_only": "#4a3aa7", "regional": "#8a8983"}
    labels = {"pretrained": "Pre-trained (no loggers)", "pretrained_calibrated": "Pre-trained + calibrated on loggers",
              "fine_tuned": "Fine-tuned through the physics", "fine_tuned_calibrated": "Fine-tuned + calibrated",
              "loggers_only": "Same network, loggers only", "regional": "Regional constant from loggers"}
    titles = {"reality": "Random mismatch", "reality_systematic": "Random + systematic mismatch"}
    xs = list(N_LOGGERS)
    metrics = [("next_dry_err_days", "Next-dry error, median (days)"), ("spearman", "S&P rank correlation"),
               ("ci90_coverage", "90 % interval coverage")]
    fig, axes = plt.subplots(len(WORLD_SEEDS), 3, figsize=(12, 7), sharex=True)
    for r, wname in enumerate(WORLD_SEEDS):
        g = agg.loc[wname]
        for c, (metric, ylab) in enumerate(metrics):
            ax = axes[r, c]
            for mname in ("pretrained_calibrated", "fine_tuned", "fine_tuned_calibrated", "loggers_only", "regional"):
                gm = g.loc[mname].loc[xs, metric]
                if gm.isna().all():
                    continue
                ax.plot(xs, gm, marker="o", ms=4, lw=2, color=colors[mname], label=labels[mname])
            ax.axhline(g.loc[("pretrained", 0), metric], color=colors["pretrained"], lw=2, ls=":", label=labels["pretrained"])
            if metric == "ci90_coverage":
                ax.axhline(0.9, color="#52514e", lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(xs)
            ax.minorticks_off()
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{titles[wname]}\n{ylab}")
            else:
                ax.set_ylabel(ylab)
            if r == len(WORLD_SEEDS) - 1:
                ax.set_xlabel("Logger fields")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("KGML-Paddy v0: pre-trained on simulator v2, tested on mismatched realities", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figures" / "fig8_kgml_paddy_v0.png", dpi=150)


if __name__ == "__main__":
    main()
