"""Experiment 17: retrain KGML-Paddy with farmers who do not log.

v1 was pre-trained only on farmers who log (70-100 % of irrigations). For a farmer who logs nothing, it
sees no irrigation, concludes the water lasts longer, and underestimates the field's S&P by about 60 %
(Experiment 16). v1b is the same network and training, except that in every batch about 30 % of the
simulated fields have their farmer log removed (kp.pretrain_v1(p_nolog=0.3)).

On the Experiment 11 test sets, with half the farmers logging (a fixed random half), v1 and v1b are
compared for logging and non-logging fields separately:
  season-end effective S&P (bias, % error, next-dry error, rank correlation);
  forecasts from days 45, 55, 65 (data before that day only): ponded today, next-dry error (all
  in-season fields), the irrigate-by-tomorrow call, 90 % coverage;
  and v1b followed by the tubewell meter correction of the non-logging fields (Experiment 16,
  simulator-referenced, command areas of 25 fields).

Outputs: results/kgml_paddy_v1b.json, results/models/kgml_paddy_v1b_pretrained.pt,
results/figures/fig16_train_with_nonloggers.png. Runtime about 15 minutes (pre-training about 6).
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
sys.path.insert(0, str(ROOT / "scripts"))
m9 = __import__("09_kgml_paddy_v0")
m11 = __import__("11_kgml_paddy_v1_state")
m16 = __import__("16_tubewell_constraint")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
WORLDS = ("prior", "reality", "reality_systematic")
P_NOLOG = 0.3
SHARE_LOGGING = 0.5
DAYS = m11.FORECAST_DAYS
K = 100
CH = kp.CH


def load(name, d, inp):
    path = OUT / "models" / f"kgml_paddy_{name}_pretrained.pt"
    model = kp.PaddyNet()
    if path.exists():
        model.load_state_dict(torch.load(path))
        return model
    t = time.time()
    model = kp.pretrain_v1(d["train"], d["val"], epochs=30, p_nolog=P_NOLOG, log=lambda s: print(s, flush=True))
    torch.save(model.state_dict(), path)
    print(f"pre-trained {name} in {time.time() - t:.0f} s", flush=True)
    return model


def season_metrics(model, te, X, inp, et_mid):
    th, st = kp.predict_v1(model, X, te["static"], X.shape[1], K)
    sp = np.median(ps.sp_effective(*(th[..., i] for i in range(4)), te["tp_abs"][None, :], inp, dt=0.05), 0)
    return sp, th, st


def group_sp(sp, te, sel, et_mid):
    y = te["sp_eff"][sel]
    m = m9.point_metrics(sp[sel], y, et_mid)
    return {"sp_log_bias": float(np.median(np.log(sp[sel] / y))), "sp_next_dry_err_days": m["next_dry_err_days"],
            "sp_median_abs_pct_err": m["median_abs_pct_err"], "sp_spearman": float(spearmanr(sp[sel], y).correlation)}


def forecast_group(model, te, X, inp, groups):
    rows = {g: [] for g in groups}
    for day in DAYS:
        H = te["H"][:, day - 1]
        th, st = kp.predict_v1(model, X, te["static"], day, K)
        for g, sel in groups.items():
            F = np.flatnonzero(sel & np.isfinite(H) & (X[:, day - 1, CH["season"]] > 0))
            t0 = inp["s0"] + day
            tp = te["tp_abs"][F]
            truth = kp.forecast_dry(te["theta"][F], H[F], t0, tp, inp, et_f=te["et_f"][F], g=te["g"][F])
            h_s = kp.sample_depth(st, day - 1, K, np.random.default_rng(day), fields=F)
            D = kp.forecast_dry(th[:, F], h_s, t0, tp[None, :], inp)
            lo, med, hi = np.quantile(D, 0.05, 0), np.median(D, 0), np.quantile(D, 0.95, 0)
            err = np.abs(med - truth)
            rows[g].append({"ponded_today_bacc": m11.bacc(H[F] > 0, st["p_wet"][F, day - 1] > 0.5),
                            "next_dry_err_days": float(np.median(err)),
                            "irrigate_by_tomorrow_bacc": m11.bacc(truth <= 1, med <= 1),
                            "ci90_coverage": float(((lo <= truth + 0.05) & (truth - 0.05 <= hi)).mean())})
    return {g: pd.DataFrame(r).mean().to_dict() for g, r in rows.items()}


def meter_correct(model, te, X, sp, th, st, logs, inp, k_ref):
    """Experiment 16's non-logger correction (simulator-referenced), command areas of 25 fields."""
    n = len(sp)
    rng = np.random.default_rng(7)
    area = rng.uniform(0.1, 0.3, n)
    tw = np.arange(n) // m16.TW_SIZE
    n_tw = tw.max() + 1
    V = np.bincount(tw, area * te["irr_true_mm"]) * (1 + rng.uniform(0.05, 0.15, n_tw)) * rng.lognormal(0, 0.05, n_tw)
    non_sp, sp_sum = m16.water_terms({**te, "X": X}, np.median(th, 0), st, inp)
    V_field = V / (1 + m16.ASSUMED_CONVEYANCE)
    fixed = np.bincount(tw, area * non_sp) + k_ref * np.bincount(tw, area * sp_sum * logs)
    denom = np.bincount(tw, area * sp_sum * ~logs)
    k = np.where(denom > 0, np.clip((V_field - fixed) / np.maximum(denom, 1e-9), 0.2, 6.0), 1.0)
    return np.where(logs, sp, sp * k[tw]), float(np.median(k))


def accounting_reference(model, d, inp):
    """kappa of the model-based accounting in the training world with everyone logging (Experiment 16)."""
    te = d["test/prior"]
    X = te["X"]
    th, st = kp.predict_v1(model, X, te["static"], X.shape[1], K)
    n = len(X)
    rng = np.random.default_rng(7)
    area = rng.uniform(0.1, 0.3, n)
    tw = np.arange(n) // m16.TW_SIZE
    n_tw = tw.max() + 1
    V = np.bincount(tw, area * te["irr_true_mm"]) * (1 + rng.uniform(0.05, 0.15, n_tw)) * rng.lognormal(0, 0.05, n_tw)
    non_sp, sp_sum = m16.water_terms(te, np.median(th, 0), st, inp)
    k = (V / (1 + m16.ASSUMED_CONVEYANCE) - np.bincount(tw, area * non_sp)) / np.bincount(tw, area * sp_sum)
    return float(np.median(k))


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    models = {"v1": load("v1", d, inp), "v1b": load("v1b", d, inp)}
    refs = {name: accounting_reference(m, d, inp) for name, m in models.items()}
    print("accounting references:", {k: round(v, 3) for k, v in refs.items()}, flush=True)
    summary = {"accounting_reference": refs}
    for w in WORLDS:
        te = d[f"test/{w}"]
        logs = np.random.default_rng(50).random(len(te["X"])) < SHARE_LOGGING
        X = kp.drop_logs(te["X"], ~logs)
        groups = {"loggers": logs, "non_loggers": ~logs}
        res = {}
        for name, model in models.items():
            t = time.time()
            sp, th, st = season_metrics(model, te, X, inp, et_mid)
            fc = forecast_group(model, te, X, inp, groups)
            res[name] = {g: {**group_sp(sp, te, sel, et_mid), **fc[g]} for g, sel in groups.items()}
            sp_c, k_med = meter_correct(model, te, X, sp, th, st, logs, inp, refs[name])
            res[f"{name}+meter"] = {g: group_sp(sp_c, te, sel, et_mid) for g, sel in groups.items()}
            res[f"{name}+meter"]["kappa_median"] = k_med
            print(f"{w} {name} ({time.time() - t:.0f} s)", flush=True)
        summary[w] = res
        print(f"== {w}")
        for name, r in res.items():
            for g in ("loggers", "non_loggers"):
                o = r[g]
                print(f"  {name:10s} {g:11s} S&P bias {o['sp_log_bias']:+.2f}, err {o['sp_next_dry_err_days']:.2f} d, "
                      f"rho {o['sp_spearman']:.2f}" +
                      (f" | ponded-today {o['ponded_today_bacc']:.3f}, next-dry {o['next_dry_err_days']:.2f} d, "
                       f"irrigate-call {o['irrigate_by_tomorrow_bacc']:.3f}, cov {o['ci90_coverage']:.2f}"
                       if "ponded_today_bacc" in o else ""), flush=True)
    (OUT / "kgml_paddy_v1b.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    series = [("v1", "#8a8983", "v1 (trained on loggers only)"), ("v1b", "#2a78d6", "v1b (30 % non-loggers in training)"),
              ("v1b+meter", "#eb6834", "v1b + tubewell meter (non-loggers)")]
    titles = {"prior": "Training\nworld", "reality": "Random\nmismatch", "reality_systematic": "Random +\nsystematic"}
    panels = [("sp_log_bias", "S&P bias (median log ratio)", False), ("sp_next_dry_err_days", "S&P: next-dry error (days)", False),
              ("ponded_today_bacc", "Ponded today? (balanced acc.)", True), ("next_dry_err_days", "Forecast next-dry error (days)", True)]
    x = np.arange(len(WORLDS))
    fig, axes = plt.subplots(2, len(panels), figsize=(14, 6.4))
    for r, g in enumerate(("non_loggers", "loggers")):
        for c, (key, lab, forecast_only) in enumerate(panels):
            ax = axes[r, c]
            for j, (name, col, label) in enumerate(series):
                if forecast_only and name.endswith("+meter"):
                    continue
                ys = [summary[w][name][g][key] for w in WORLDS]
                ax.scatter(x + (j - 1) * 0.2, ys, color=col, s=45, label=label, zorder=3)
            if key == "sp_log_bias":
                ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([titles[w] for w in WORLDS], fontsize=8)
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(lab, fontsize=9)
            if c == 0:
                ax.set_ylabel("Fields that do not log" if g == "non_loggers" else "Fields that log", fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Training KGML-Paddy with farmers who do not log (half the test farmers log)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig16_train_with_nonloggers.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "kgml_paddy_v1b.json").read_text()))
    else:
        main()
