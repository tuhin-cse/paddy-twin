"""Experiment 8: simulator mismatch, and how many loggers fine-tuning needs.

The first test (KGML-Paddy, following Liu et al. 2024, Nat. Commun.).
Every earlier S&P result trained and tested on the same simulator. Here the
estimator is pre-trained on that simple simulator (constant S&P per field, as in
04b), but tested on a richer "reality" whose physics it has never seen:

    SP(h, tau) = a (1 + w tau/80) exp(-c tau)  +  (a/Lp + s_lat) h
                 base rate, water table falls,     head dependence: Darcy
                 puddled layer seals               through the puddled layer
                                                   (thickness Lp) + bund seepage

with tau = days since transplanting, h = ponded depth (mm). At the "strong" level
30 % of farmers also top up before the field dries (as in 04c). "strong+obs" adds
the two observation mismatches the campaign will certainly meet: satellite
detection worse than simulated (recall -0.10, specificity -0.05 in every LAI
bin), and pump hours converted to depth with a per-field bias (lognormal,
sigma 0.25; pump discharge and field area are not known exactly).

The truth a logger would give is the effective S&P of a 50 mm irrigation on
1 March (about LAI 2): SP_eff = (50 + rain) / D - mean(Kc ET0) over the D days
the water lasts. Estimators, all from the same observation summaries as 04b:

    regional   median SP_eff of the n logger fields (5 mm/d if n = 0)
    pure ML    gradient boosting trained only on the n logger fields
    pretrained quantile boosting trained on 8,000 simple-simulator fields (= 04b)
    fine-tune  pretrained + shrinkage recalibration log y = alpha + beta p
               (prior alpha ~ 0, beta ~ 1: "fine-tune the last layer")
    fine-tune+ the same plus a shallow residual booster on features (n >= 20)
    physics    per-field closed-form posterior of hydro.py (constant-S&P model,
               no learning; the per-field Bayesian calibration baseline)

for n = 0 ... 160 logger fields drawn from a pool of 2,000 rich fields, repeated
over random draws, tested on 1,000 other rich fields. Scenarios: S0 (satellite
only) and S2 (satellite + app irrigation date + pump hours). No SAR.

Outputs results/simulator_mismatch.json, results/simulator_mismatch_runs.csv.gz
and results/figures/fig7_simulator_mismatch.png. Runtime about 15 minutes.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sbi = __import__("04b_simulation_based_inference")
tw = sbi.tw
hydro = tw.hydro
OUT = tw.OUT
FIG = OUT / "figures"

SCEN = {"S0 satellite only": (False, False, False),
        "S2 satellite + app date & pump hours": (True, True, False)}
LEVELS = {  # ranges of the rich-physics parameters; "none" is the simple simulator itself
    "none":       dict(Lp=None,       s_lat=0.0,  c=0.0,   w=0.0, topup=0.0, obs_mismatch=False),
    "moderate":   dict(Lp=(150, 400), s_lat=0.02, c=0.005, w=0.2, topup=0.0, obs_mismatch=False),
    "strong":     dict(Lp=(60, 200),  s_lat=0.05, c=0.012, w=0.5, topup=0.3, obs_mismatch=False),
    "strong+obs": dict(Lp=(60, 200),  s_lat=0.05, c=0.012, w=0.5, topup=0.3, obs_mismatch=True),
}
PUMP_BIAS_SIGMA = 0.25
N_LOGGERS = {0: 1, 2: 50, 5: 30, 10: 20, 20: 10, 40: 10, 80: 5, 160: 5}
N_POOL, N_TEST, N_PRETRAIN = 2000, 1000, 8000
T_MID = tw.dayof("2024-03-01")
KC2 = hydro.kc_rice(2.0)
ET_MID = float(tw.wx.loc["2024-02-01":"2024-04-15"].et0.mean())


def draw_physics(rng, level):
    p = LEVELS[level]
    a = float(np.clip(rng.lognormal(np.log(5.0), 0.6), 0.5, 30))  # same prior as the simple simulator
    if p["Lp"] is None:
        return dict(a=a, b=0.0, c=0.0, w=0.0)
    return dict(a=a, b=a / rng.uniform(*p["Lp"]) + rng.uniform(0, p["s_lat"]),
                c=rng.uniform(0, p["c"]), w=rng.uniform(0, p["w"]))


def sp_fn(ph, a_season):
    return lambda h, tau: a_season * (1 + ph["w"] * tau / 80) * np.exp(-ph["c"] * tau) + ph["b"] * max(h, 0.0)


def sp_effective(ph, a_season, transplant):
    """What a logger shows: effective S&P of a 50 mm irrigation on 1 March.

    Used as the truth at every level, and as the pre-training target (for the
    simple simulator it equals the season's constant S&P).
    """
    f, dt = sp_fn(ph, a_season), 0.01
    days = np.arange(T_MID, T_MID + 41)
    etc = (hydro.kc_rice(tw.lai_curve(days + 0.5, transplant)) * tw.ET0[days]).tolist()
    rain_d = tw.RAIN[days].tolist()
    t, h, loss_et, rain = 0.0, 50.0, 0.0, 0.0
    while h > 0 and t < 40:
        d = int(t)
        h += (rain_d[d] - etc[d] - f(h, T_MID + t - transplant)) * dt
        loss_et += etc[d] * dt
        rain += rain_d[d] * dt
        t += dt
    return (50 + rain) / t - loss_et / t


def degraded(rates):
    """Real detection worse than the PROSAIL-derived rates."""
    lo, rec, spec = rates
    return lo, np.clip(rec - 0.10, 0.5, 1), np.clip(spec - 0.05, 0.5, 1)


def build(level, n_fields, seed0, obs):
    rows, seasons = [], {}
    for n in range(n_fields):
        rng = np.random.default_rng(seed0 + n)
        ph = draw_physics(rng, level)
        a_season = ph["a"] * (1 + 0.1 * rng.standard_normal())  # year-to-year puddling, as in 04/04b
        lv = LEVELS[level]
        obs_l = (*obs[:2], degraded(obs[2])) if lv["obs_mismatch"] else obs
        s = tw.simulate_season(rng, sp_fn(ph, a_season) if level != "none" else a_season,
                               *obs_l, sar=False, topup_frac=lv["topup"])
        if lv["obs_mismatch"]:  # logged depth = true depth x per-field pump-rating bias
            s = {**s, "irr_h": s["irr_h"] * rng.lognormal(0, PUMP_BIAS_SIGMA)}
        y = sp_effective(ph, a_season, s["transplant"])
        seasons[n] = s
        for name, flags in SCEN.items():
            x = sbi.features([s], *flags, np.random.default_rng(seed0 + 900_000 + n))
            rows.append({"field": n, "scenario": name, "y": y, **ph, **x})
    return pd.DataFrame(rows), seasons


def physics_baseline(seasons, flags):
    """Closed-form per-field posterior (hydro.py), constant S&P; no learning."""
    out = []
    for n in sorted(seasons):
        est, _ = tw.fit_field([seasons[n]], *flags, np.random.default_rng(30_000 + n))
        out.append(est["median"] if est else np.nan)
    return np.array(out)


def gbm(**kw):
    return HistGradientBoostingRegressor(random_state=0, **kw)


def recalibrate(p, y, s=0.3, ta=0.5, tb=0.3):
    """MAP of log y = alpha + beta p, priors alpha ~ N(0, ta), beta ~ N(1, tb)."""
    X = np.c_[np.ones_like(p), p]
    P0 = np.diag([1 / ta**2, 1 / tb**2])
    A = X.T @ X / s**2 + P0
    rhs = X.T @ y / s**2 + P0 @ np.array([0.0, 1.0])
    return np.linalg.solve(A, rhs)


def metrics(y, yhat, lo=None, hi=None):
    D = 50 / (KC2 * ET_MID + y)
    err = np.abs(50 / (KC2 * ET_MID + np.clip(yhat, 0, None)) - D)
    m = {"spearman": float(spearmanr(y, yhat).correlation) if np.ptp(yhat) > 1e-9 else 0.0,  # constant: no ranking
         "median_abs_pct_err": float(np.median(np.abs(yhat - y) / y) * 100),
         "next_dry_err_days": float(np.median(err)),
         "share_err_gt1d": float((err > 1).mean())}
    if lo is not None:
        m["ci90_coverage"] = float(((lo <= y) & (y <= hi)).mean())
    return m


def main():
    real = json.loads((OUT / "real_validation_metrics.json").read_text())
    season = pd.read_csv(OUT / "real_season_curve.csv", index_col="date")
    obs_days = np.array([tw.dayof(d) for d in real["dates"]], float)
    obs_valid = np.array([season.valid_rice_frac.get(d, 0.2) for d in real["dates"]])
    obs = (obs_days, obs_valid, tw.detection_rates())

    cache = ROOT / "data" / "cache" / "simulator_mismatch_sets.pkl"
    if cache.exists():
        sets = pd.read_pickle(cache)
    else:
        sets = {"pretrain": build("none", N_PRETRAIN, 3_000_000, obs)[0]}
        for i, lvl in enumerate(LEVELS):
            pool, _ = build(lvl, N_POOL, 4_000_000 + i * 100_000, obs)
            test, test_seasons = build(lvl, N_TEST, 5_000_000 + i * 100_000, obs)
            phys = {name: physics_baseline(test_seasons, flags) for name, flags in SCEN.items()}
            sets[lvl] = (pool, test, phys)
            print(f"simulated {lvl}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(sets, cache)

    runs, summary = [], {}
    for name in SCEN:
        tr = sets["pretrain"][sets["pretrain"].scenario == name]
        cols = [c for c in tr.columns if c not in ("field", "scenario", "y", "a", "b", "c", "w")
                and tr[c].notna().mean() > 0.5]
        pre = {q: gbm(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05).fit(tr[cols], np.log(tr.y))
               for q in (0.05, 0.5, 0.95)}
        for lvl in LEVELS:
            pool, test, phys = sets[lvl]
            pool, test = pool[pool.scenario == name].reset_index(drop=True), test[test.scenario == name].reset_index(drop=True)
            y_te, ly_pool = test.y.to_numpy(), np.log(pool.y.to_numpy())
            p_te = {q: m.predict(test[cols]) for q, m in pre.items()}
            p_pool = pre[0.5].predict(pool[cols])
            summary.setdefault(name, {})[lvl] = {
                "sp_eff_median": float(np.median(y_te)),
                "sp_eff_p10_p90": [float(np.percentile(y_te, 10)), float(np.percentile(y_te, 90))],
                "physics_closed_form": metrics(y_te, np.nan_to_num(phys[name], nan=5.0)),
            }
            for n, reps in N_LOGGERS.items():
                for r in range(reps):
                    idx = np.random.default_rng(7_000 + 100 * n + r).choice(len(pool), n, replace=False)
                    ly = ly_pool[idx]
                    reg = np.exp(np.median(ly)) if n else 5.0
                    res = {"regional": metrics(y_te, np.full_like(y_te, reg)),
                           "pretrained": metrics(y_te, np.exp(p_te[0.5]), np.exp(p_te[0.05]), np.exp(p_te[0.95]))}
                    if n >= 5:
                        ml = gbm(max_iter=200, learning_rate=0.05, min_samples_leaf=max(2, n // 10))
                        ml.fit(pool.loc[idx, cols], ly)
                        res["pure_ml"] = metrics(y_te, np.exp(ml.predict(test[cols])))
                    else:
                        res["pure_ml"] = res["regional"]
                    if n:
                        al, be = recalibrate(p_pool[idx], ly)
                        f_te = al + be * p_te[0.5]
                        if n >= 10:  # interval from logger residuals (Gaussian in log space)
                            rr = ly - (al + be * p_pool[idx])
                            half = 1.645 * np.sqrt(np.mean(rr**2) * n / (n - 2))
                            lo, hi = f_te - half, f_te + half
                        else:
                            lo, hi = al + be * p_te[0.05], al + be * p_te[0.95]
                        res["fine_tune"] = metrics(y_te, np.exp(f_te), np.exp(lo), np.exp(hi))
                        if n >= 20:
                            Xl = pool.loc[idx, cols].assign(p=p_pool[idx])
                            boost = gbm(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=max(3, n // 8))
                            boost.fit(Xl, ly - (al + be * p_pool[idx]))
                            res["fine_tune_plus"] = metrics(y_te, np.exp(f_te + boost.predict(test[cols].assign(p=p_te[0.5]))))
                        else:
                            res["fine_tune_plus"] = res["fine_tune"]
                    else:
                        res["fine_tune"] = res["fine_tune_plus"] = res["pretrained"]
                    for model, m in res.items():
                        runs.append({"scenario": name, "level": lvl, "n_loggers": n, "draw": r, "model": model, **m})
            print(f"done {name} / {lvl}", flush=True)

    df = pd.DataFrame(runs)
    df.to_csv(OUT / "simulator_mismatch_runs.csv.gz", index=False)
    agg = df.groupby(["scenario", "level", "model", "n_loggers"]).agg(
        spearman=("spearman", "mean"), spearman_p10=("spearman", lambda s: s.quantile(0.1)),
        spearman_p90=("spearman", lambda s: s.quantile(0.9)), next_dry_err_days=("next_dry_err_days", "mean"),
        share_err_gt1d=("share_err_gt1d", "mean"), median_abs_pct_err=("median_abs_pct_err", "mean"),
        ci90_coverage=("ci90_coverage", "mean")).reset_index()
    for (name, lvl, model), g in agg.groupby(["scenario", "level", "model"]):
        vals = g.drop(columns=["scenario", "level", "model"]).set_index("n_loggers")
        summary[name][lvl][model] = {int(n): {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
                                     for n, row in vals.iterrows()}
    (OUT / "simulator_mismatch.json").write_text(json.dumps(summary, indent=2))

    # Figure (scenario S2): rank correlation and next-dry error vs number of logger fields, per mismatch level.
    colors = {"pretrained": "#2a78d6", "fine_tune": "#eb6834", "fine_tune_plus": "#1baf7a",
              "pure_ml": "#4a3aa7", "regional": "#8a8983"}
    labels = {"pretrained": "Pre-trained on simple simulator", "fine_tune": "Pre-trained + fine-tuned",
              "fine_tune_plus": "Fine-tuned + residual booster", "pure_ml": "Pure ML on loggers only",
              "regional": "Regional constant from loggers"}
    s2name = "S2 satellite + app date & pump hours"
    s2 = agg[agg.scenario == s2name]
    xs = [n for n in N_LOGGERS if n > 0]
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(3.2 * len(LEVELS), 6.4), sharey="row", sharex=True)
    rows = [("spearman", "S&P rank correlation", ("pretrained", "fine_tune", "fine_tune_plus", "pure_ml")),
            ("next_dry_err_days", "Next-dry error, median (days)",
             ("pretrained", "fine_tune", "fine_tune_plus", "pure_ml", "regional"))]
    for r, (metric, ylab, models) in enumerate(rows):
        for c, lvl in enumerate(LEVELS):
            ax = axes[r, c]
            g = s2[(s2.level == lvl) & (s2.n_loggers > 0)]
            for model in models:
                gm = g[g.model == model].sort_values("n_loggers")
                top = model == "pretrained"  # drawn dotted on top: recalibration keeps its ranks, so lines coincide
                ax.plot(gm.n_loggers, gm[metric], marker="o", ms=4, lw=2, color=colors[model], label=labels[model],
                        ls=":" if top else "-", zorder=5 if top else 2)
            phys = summary[s2name][lvl]["physics_closed_form"][metric]
            ax.axhline(phys, color="#52514e", lw=1, ls="--", label="Per-field physics calibration (no learning)")
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(xs)
            ax.minorticks_off()
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if r == 0:
                ax.set_title(f"Mismatch: {lvl}", fontsize=10)
            else:
                ax.set_xlabel("Logger fields used for training")
            if c == 0:
                ax.set_ylabel(ylab)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Satellite + app (date, pump hours): pre-trained on the simple simulator, tested on richer reality",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig7_simulator_mismatch.png", dpi=150)

    view = s2[s2.n_loggers.isin([0, 5, 10, 20, 40, 160])].pivot_table(
        index=["level", "model"], columns="n_loggers", values="spearman").round(2)
    print(view.to_string())


if __name__ == "__main__":
    main()
