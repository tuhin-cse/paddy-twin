"""Experiment 10: fair tree baseline for KGML-Paddy v0 on simulator v2.

Experiment 9 found the sequence network far more robust to mismatch (0.46 d) than Experiment 8's
gradient-boosted trees (0.84 d), but the two ran on different simulators, with different
observations and training-set sizes. This script trains quantile gradient boosting on exactly the
Experiment 9 data (cache of scripts/09: 20,000 prior-world training fields; the same test worlds,
truth and logger subsets) so the network's contribution can be separated from the simulator's.

Tree models (quantiles 0.05 / 0.5 / 0.95 of log effective S&P):
  trees_04b   the summary features of scripts/04b (via its features()), rebuilt from the daily
              arrays: what Experiments 4b and 8 used;
  trees_rich  04b features plus what the network also sees: the 8 field attributes, Sentinel-2
              and Sentinel-1 counts and wet fractions separately, wet fraction by observed-LAI
              band and by two-week window, logged irrigation count and depth, transplanting day,
              season ET0 and rain.

With n logger fields (same draws as Experiment 9), for trees_rich:
  calibrated  split-conformal scaling of the 90 % interval against each logger field's own
              effective S&P (awdrt.kgml_paddy.logger_sp_eff), as for the network;
  fine-tuned  Experiment 8's recalibration + residual booster (booster from 20 loggers), fitted
              to the logger estimates, not the truth; then calibrated as above.

Outputs: results/tree_baseline_v2.json, results/figures/fig9_network_vs_trees.png.
Needs the Experiment 9 cache (run scripts/09 first). Runtime about 10 minutes.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
m9 = __import__("09_kgml_paddy_v0")
m8 = __import__("08_simulator_mismatch")
sbi = m8.sbi
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CH = {c: i for i, c in enumerate(ps.CHANNELS)}
WORLDS = list(m9.WORLD_SEEDS)


def seasons_04b(X, tp_abs, s0):
    """Rebuild 04b 'season' dicts (absolute day index, as in the twin) from the daily arrays."""
    out = []
    for i in range(len(X)):
        x = X[i]
        wet = (x[:, CH["s2_wet"]] + x[:, CH["s1_wet"]] + x[:, CH["nisar_wet"]]) > 0
        dry = (x[:, CH["s2_dry"]] + x[:, CH["s1_dry"]] + x[:, CH["nisar_dry"]]) > 0
        obs = np.flatnonzero(wet | dry)
        irr = np.flatnonzero(x[:, CH["irr_logged"]] > 0)
        out.append(dict(days=(obs + s0 + 0.5).astype(float), state=wet[obs].astype(int),
                        irr_t=(irr + s0 + 0.5).astype(float), irr_h=x[irr, CH["depth_logged"]] * 100,
                        gone_t=(np.flatnonzero(x[:, CH["gone"]] > 0) + s0 + 0.5).astype(float), transplant=tp_abs[i]))
    return out


def features(data, s0, rich):
    X = data["X"]
    rows = [sbi.features([s], True, True, False, np.random.default_rng(i))
            for i, s in enumerate(seasons_04b(X, data["tp_abs"], s0))]
    f = pd.DataFrame(rows)
    if not rich:
        return f
    season = X[:, :, CH["season"]] > 0
    lai = X[:, :, CH["lai_obs"]] * 5
    for sensor in ("s2", "s1"):
        w, d = X[:, :, CH[f"{sensor}_wet"]], X[:, :, CH[f"{sensor}_dry"]]
        n = w.sum(1) + d.sum(1)
        f[f"n_{sensor}"] = n
        f[f"wet_frac_{sensor}"] = np.where(n > 0, w.sum(1) / np.maximum(n, 1), np.nan)
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 9)):
            band = (lai >= lo) & (lai < hi)
            nb = ((w + d) * band).sum(1)
            f[f"wet_frac_{sensor}_lai{lo}"] = np.where(nb > 0, (w * band).sum(1) / np.maximum(nb, 1), np.nan)
    w_all = X[:, :, CH["s2_wet"]] + X[:, :, CH["s1_wet"]]
    o_all = w_all + X[:, :, CH["s2_dry"]] + X[:, :, CH["s1_dry"]]
    for k in range(0, X.shape[1], 14):
        n = o_all[:, k:k + 14].sum(1)
        f[f"wet_frac_wk{k // 7}"] = np.where(n > 0, w_all[:, k:k + 14].sum(1) / np.maximum(n, 1), np.nan)
    f["n_irr_logged"] = X[:, :, CH["irr_logged"]].sum(1)
    f["depth_logged_total"] = X[:, :, CH["depth_logged"]].sum(1) * 100
    f["tp_day"] = data["tp"]
    f["season_days"] = season.sum(1)
    f["et0_season"] = (X[:, :, CH["et0"]] * season).sum(1) * 10 / np.maximum(season.sum(1), 1)
    f["rain_season"] = (X[:, :, CH["rain"]] * season).sum(1) * 20
    for j in range(data["static"].shape[1]):
        f[f"attr{j}"] = data["static"][:, j]
    return f


def fit_quantiles(F, y):
    return {q: m8.gbm(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05).fit(F, np.log(y))
            for q in (0.05, 0.5, 0.95)}


def conformal_q(med, lo, hi, est):
    hw = (np.log(hi) - np.log(lo)) / 2
    ok = np.isfinite(est) & (est > 0)
    n = ok.sum()
    if n < 5:
        return 1.0
    score = np.abs(np.log(est[ok]) - np.log(med[ok])) / hw[ok]
    return float(np.quantile(score, min(1.0, np.ceil((n + 1) * 0.9) / n)))


def scaled(med, lo, hi, q):
    hw = (np.log(hi) - np.log(lo)) / 2
    return med * np.exp(-q * hw), med * np.exp(q * hw)


def main():
    inp = ps.load_inputs()
    s0 = inp["s0"]
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m9.sets(inp)
    net = json.loads((OUT / "kgml_paddy_v0.json").read_text())

    feats = {}
    for rich in (False, True):
        tag = "trees_rich" if rich else "trees_04b"
        feats[tag] = {k: features(d[k], s0, rich) for k in ["train", "prior_test"] + [f"{w}/{p}" for w in WORLDS
                                                                                       for p in ("pool", "test")]}
        print(f"features {tag}: {feats[tag]['train'].shape[1]}", flush=True)

    summary, runs = {}, []
    for tag, F in feats.items():
        cols = [c for c in F["train"].columns if F["train"][c].notna().mean() > 0.5]
        mdl = fit_quantiles(F["train"][cols], d["train"]["sp_eff"])
        pred = lambda k: {q: np.exp(m.predict(F[k][cols])) for q, m in mdl.items()}  # noqa: E731
        p = pred("prior_test")
        summary[f"{tag}|prior_test"] = m9.point_metrics(p[0.5], d["prior_test"]["sp_eff"], et_mid, p[0.05], p[0.95])
        for w in WORLDS:
            pool, test = d[f"{w}/pool"], d[f"{w}/test"]
            pt, pp = pred(f"{w}/test"), pred(f"{w}/pool")
            m0 = m9.point_metrics(pt[0.5], test["sp_eff"], et_mid, pt[0.05], pt[0.95])
            summary[f"{tag}|{w}"] = m0
            runs.append({"world": w, "model": tag, "n_loggers": 0, "draw": 0, **m0})
            print(f"{tag} {w}: {m0['next_dry_err_days']:.2f} d, rho {m0['spearman']:.2f}, cov {m0['ci90_coverage']:.2f}",
                  flush=True)
            if tag != "trees_rich":
                continue
            est_pool = kp.logger_sp_eff(pool, inp)
            has_rec = np.isfinite(pool["r"]).sum(1) >= 5
            X_pool, X_test = F[f"{w}/pool"][cols], F[f"{w}/test"][cols]
            for n, reps in m9.N_LOGGERS.items():
                for r_ in range(reps):
                    idx = np.random.default_rng(100 * n + r_).choice(np.flatnonzero(has_rec), n, replace=False)
                    est = est_pool[idx]
                    q = conformal_q(pp[0.5][idx], pp[0.05][idx], pp[0.95][idx], est)
                    lo, hi = scaled(pt[0.5], pt[0.05], pt[0.95], q)
                    row = {"world": w, "n_loggers": n, "draw": r_, "conformal_scale": q}
                    runs.append({**row, "model": "trees_rich_calibrated",
                                 **m9.point_metrics(pt[0.5], test["sp_eff"], et_mid, lo, hi)})
                    ok = np.isfinite(est) & (est > 0)
                    lp_pool, ly = np.log(pp[0.5][idx][ok]), np.log(est[ok])
                    al, be = m8.recalibrate(lp_pool, ly)
                    f_te = al + be * np.log(pt[0.5])
                    if ok.sum() >= 20:
                        boost = m8.gbm(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=max(3, ok.sum() // 8))
                        boost.fit(X_pool.iloc[idx[ok]].assign(p=lp_pool), ly - (al + be * lp_pool))
                        f_te = f_te + boost.predict(X_test.assign(p=np.log(pt[0.5])))
                    med = np.exp(f_te)
                    lo, hi = scaled(med, med * pt[0.05] / pt[0.5], med * pt[0.95] / pt[0.5], q)
                    runs.append({**row, "model": "trees_rich_fine_tuned_calibrated",
                                 **m9.point_metrics(med, test["sp_eff"], et_mid, lo, hi)})

    df = pd.DataFrame(runs)
    agg = df.groupby(["world", "model", "n_loggers"]).mean(numeric_only=True).drop(columns="draw")
    summary["with_loggers"] = {f"{w}|{m}|{n}": {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
                               for (w, m, n), row in agg.iterrows()}
    (OUT / "tree_baseline_v2.json").write_text(json.dumps(summary, indent=2))

    # comparison table printed for the write-up
    print("\nno loggers: next-dry days / rank corr / coverage")
    print(f"  prior_test  network {net['pretrained_in_distribution']['next_dry_err_days']:.2f} / "
          f"{net['pretrained_in_distribution']['spearman']:.2f} / {net['pretrained_in_distribution']['ci90_coverage']:.2f}")
    for tag in feats:
        m = summary[f"{tag}|prior_test"]
        print(f"  prior_test  {tag} {m['next_dry_err_days']:.2f} / {m['spearman']:.2f} / {m['ci90_coverage']:.2f}")
    for w in WORLDS:
        nm = net["by_world_model_loggers"][f"{w}|pretrained|0"]
        print(f"  {w}  network {nm['next_dry_err_days']:.2f} / {nm['spearman']:.2f} / {nm['ci90_coverage']:.2f}")
        for tag in feats:
            m = summary[f"{tag}|{w}"]
            print(f"  {w}  {tag} {m['next_dry_err_days']:.2f} / {m['spearman']:.2f} / {m['ci90_coverage']:.2f}")
    print(agg[["next_dry_err_days", "spearman", "ci90_coverage", "median_abs_pct_err"]].round(3).to_string())

    # figure: network vs rich trees, calibrated and fine-tuned + calibrated, per world
    xs = list(m9.N_LOGGERS)
    series = {("network", "pretrained_calibrated"): ("KGML-Paddy v0, calibrated", "#2a78d6", "-"),
              ("network", "fine_tuned_calibrated"): ("KGML-Paddy v0, fine-tuned + calibrated", "#2a78d6", "--"),
              ("trees", "trees_rich_calibrated"): ("Trees (rich features), calibrated", "#eb6834", "-"),
              ("trees", "trees_rich_fine_tuned_calibrated"): ("Trees (rich features), fine-tuned + calibrated", "#eb6834", "--")}
    titles = {"reality": "Random mismatch", "reality_systematic": "Random + systematic mismatch"}
    fig, axes = plt.subplots(len(WORLDS), 2, figsize=(9, 6.6), sharex=True)
    for r, w in enumerate(WORLDS):
        for c, (metric, ylab) in enumerate([("next_dry_err_days", "Next-dry error, median (days)"),
                                            ("ci90_coverage", "90 % interval coverage")]):
            ax = axes[r, c]
            for (fam, mname), (label, col, ls) in series.items():
                if fam == "network":
                    ys = [net["by_world_model_loggers"][f"{w}|{mname}|{n}"][metric] for n in xs]
                else:
                    ys = [agg.loc[(w, mname, n), metric] for n in xs]
                ax.plot(xs, ys, marker="o", ms=4, lw=2, color=col, ls=ls, label=label)
            ax.axhline(net["by_world_model_loggers"][f"{w}|pretrained|0"][metric], color="#2a78d6", lw=1, ls=":",
                       label="KGML-Paddy v0, no loggers")
            ax.axhline(summary[f"trees_rich|{w}"][metric], color="#eb6834", lw=1, ls=":", label="Trees (rich), no loggers")
            if metric == "ci90_coverage":
                ax.axhline(0.9, color="#52514e", lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(xs)
            ax.minorticks_off()
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_ylabel(f"{titles[w]}\n{ylab}" if c == 0 else ylab)
            if r == len(WORLDS) - 1:
                ax.set_xlabel("Logger fields")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Same data, same truth: KGML-Paddy v0 vs gradient-boosted trees on simulator v2", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "figures" / "fig9_network_vs_trees.png", dpi=150)


if __name__ == "__main__":
    main()
