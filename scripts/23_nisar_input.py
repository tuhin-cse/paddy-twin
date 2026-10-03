"""Experiment 23: KGML-Paddy with NISAR L-band as an input (v1n).

v1b (Experiment 17) reads Sentinel-2 and Sentinel-1; NISAR was public from July 2026 but never in its
training data. v1n is v1b's recipe (20,000 fields, p_nolog 0.3, S&P law, physics loss, 30 epochs) pre-trained on
the same world with NISAR on: every 12 days at 06:00, wet/dry correct with probability 0.8 at any canopy
(simulator v2's assumption; there is no NISAR accuracy for the site yet).

Comparison on the same fields: with NISAR on, the simulator draws the same fields and the same truth and
adds NISAR passes. Each test set is therefore scored three ways:
  v1b              NISAR channels removed (what v1b was trained on);
  v1n              with NISAR;
  v1n, no NISAR    NISAR removed (an acquisition gap, or NISAR data not yet processed).
Test sets: the three Experiment 11 worlds with NISAR, and the random-mismatch world with NISAR accuracy
0.7 and 0.9 (v1n still assumes 0.8).
Metrics: season S&P (next-dry error, rank correlation, 90 % coverage); forecasts from days 45, 55, 65
(next-dry error, ponded today, irrigate-by-tomorrow, coverage); and "ponded today?" by canopy (true LAI
< 2, 2-3.5, >= 3.5) from days 35-85 every 10 days, where Sentinel-1 is weakest.

Follow-up (part 2), after part 1 found almost no gain at a 12-day revisit:
  - "On the pass day": ponded? at the end of each NISAR acquisition day (day >= 25), using data through that
    day, under the canopy (true LAI >= 2) - the most NISAR can add to today's state;
  - a 6-day revisit (ascending + descending passes): v1n6 is pre-trained with it, and v1b, v1n and v1n6 are
    scored on 6-day test sets.

Part 3 (--fusion): NISAR through an explicit Bayes update of the model's prior on the pass day, instead of
as a learned input (see fusion_check).

Outputs: results/nisar_input.json, results/figures/fig22_nisar_input.png,
results/models/kgml_paddy_v1n_pretrained.pt and kgml_paddy_v1n6_pretrained.pt. Runtime about 40 minutes
(`--plot-only` redraws the figure).
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
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m11 = __import__("11_kgml_paddy_v1_state")
m17 = __import__("17_train_with_nonloggers")
m18b = __import__("18b_ablation_followup")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "nisar_input_sets.pkl"
V1N = OUT / "models" / "kgml_paddy_v1n_pretrained.pt"
V1N6 = OUT / "models" / "kgml_paddy_v1n6_pretrained.pt"
CACHE6 = ROOT / "data" / "cache" / "nisar_input_sets_6d.pkl"
TESTS6 = {"random mismatch, 6-day NISAR": ("reality", 5, 0.8), "random + systematic, 6-day NISAR": ("reality_systematic", 7, 0.8)}
NISAR_CH = [kp.CH["nisar_wet"], kp.CH["nisar_dry"]]
TESTS = {  # name: (world, seed, NISAR accuracy)
    "training world": ("prior", 3, 0.8), "random mismatch": ("reality", 5, 0.8),
    "random + systematic": ("reality_systematic", 7, 0.8),
    "random mismatch, NISAR 0.7": ("reality", 5, 0.7), "random mismatch, NISAR 0.9": ("reality", 5, 0.9),
}
STATE_DAYS = (35, 45, 55, 65, 75, 85)
LAI_BINS = ((0, 2), (2, 3.5), (3.5, 9))


def nisar_world(name, acc=0.8, revisit=12):
    return replace(ps.WORLDS[name], nisar=True, nisar_acc=acc, nisar_revisit=revisit)


def sets6(inp):
    if CACHE6.exists():
        return pd.read_pickle(CACHE6)
    out = {"train": ps.simulate(nisar_world("prior", revisit=6), 20_000, 1, inp),
           "val": ps.simulate(nisar_world("prior", revisit=6), 2_000, 2, inp)}
    for name, (w, seed, acc) in TESTS6.items():
        out[f"test/{name}"] = ps.simulate(nisar_world(w, acc, 6), 2_000, seed, inp)
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        drop = ("logger", "h6") if k in ("train", "val") else ("logger",)
        out[k] = {kk: v for kk, v in d.items() if kk not in drop}
    pd.to_pickle(out, CACHE6)
    return out


def pass_day_canopy(model, te, X, inp, revisit):
    """Balanced accuracy of "ponded at the end of a NISAR day?" using data through that day, LAI >= 2."""
    truth, pred = [], []
    for day in range(5, X.shape[1] - 1, revisit):
        if day < 25:
            continue
        _, st = kp.predict_v1(model, X, te["static"], day + 1, 8)
        H = te["H"][:, day]
        lai = ps.lai_curve(inp["s0"] + day + 0.5, te["tp_abs"], te["g"])
        ok = np.isfinite(H) & (X[:, day, kp.CH["season"]] > 0) & (lai >= 2)
        truth.append(H[ok] > 0)
        pred.append(st["p_wet"][ok, day] > 0.5)
    return {"pass_day_bacc_canopy": m11.bacc(np.concatenate(truth), np.concatenate(pred))}


def pretrained(path, d):
    if not path.exists():
        t = time.time()
        m = kp.pretrain_v1(d["train"], d["val"], epochs=30, p_nolog=m17.P_NOLOG, log=lambda s: print(s, flush=True))
        torch.save(m.state_dict(), path)
        print(f"pre-trained {path.name} in {time.time() - t:.0f} s", flush=True)
    m = kp.PaddyNet()
    m.load_state_dict(torch.load(path))
    return m


def sets(inp):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {"train": ps.simulate(nisar_world("prior"), 20_000, 1, inp), "val": ps.simulate(nisar_world("prior"), 2_000, 2, inp)}
    for name, (w, seed, acc) in TESTS.items():
        out[f"test/{name}"] = ps.simulate(nisar_world(w, acc), 2_000, seed, inp)
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        drop = ("logger", "h6") if k in ("train", "val") else ("logger",)
        out[k] = {kk: v for kk, v in d.items() if kk not in drop}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def no_nisar(X):
    X = X.copy()
    X[:, :, NISAR_CH] = 0
    return X


def state_by_canopy(model, te, X, inp):
    """Balanced accuracy of "ponded at the end of day d-1?" using data before day d, by true LAI."""
    truth, pred, lai = [], [], []
    for day in STATE_DAYS:
        _, st = kp.predict_v1(model, X, te["static"], day, 8)
        H = te["H"][:, day - 1]
        ok = np.isfinite(H) & (X[:, day - 1, kp.CH["season"]] > 0)
        truth.append(H[ok] > 0)
        pred.append(st["p_wet"][ok, day - 1] > 0.5)
        lai.append(ps.lai_curve(inp["s0"] + day - 0.5, te["tp_abs"][ok], te["g"][ok]))
    truth, pred, lai = np.concatenate(truth), np.concatenate(pred), np.concatenate(lai)
    return {f"ponded_bacc_lai_{lo}-{hi}": m11.bacc(truth[(lai >= lo) & (lai < hi)], pred[(lai >= lo) & (lai < hi)])
            for lo, hi in LAI_BINS}


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = sets(inp)
    v1b = kp.PaddyNet()
    v1b.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1b_pretrained.pt"))
    v1n = pretrained(V1N, d)
    d6 = sets6(inp)
    v1n6 = pretrained(V1N6, d6)
    summary = {}
    runs = [(name, d[f"test/{name}"], 12, (("v1b", v1b, True), ("v1n", v1n, False), ("v1n, no NISAR", v1n, True))) for name in TESTS]
    runs += [(name, d6[f"test/{name}"], 6, (("v1b", v1b, True), ("v1n", v1n, False), ("v1n6", v1n6, False), ("v1n6, no NISAR", v1n6, True)))
             for name in TESTS6]
    for name, te, revisit, variants in runs:
        allf = {"all": np.ones(len(te["X"]), bool)}
        res = {}
        for label, model, drop in variants:
            X = no_nisar(te["X"]) if drop else te["X"]
            t = time.time()
            o = m18b.metrics(model, te, X, inp, et_mid, allf)["all"]
            o.update(state_by_canopy(model, te, X, inp))
            o.update(pass_day_canopy(model, te, X, inp, revisit))
            res[label] = o
            print(f"{name} | {label}: S&P {o['sp_next_dry_err_days']:.2f} d (rho {o['sp_spearman']:.2f}) | forecast "
                  f"{o['fc_next_dry_err_days']:.2f} d, ponded today {o['fc_ponded_today_bacc']:.3f}, irrigate "
                  f"{o['fc_irrigate_by_tomorrow_bacc']:.3f}, cov {o['fc_ci90_coverage']:.2f} | ponded by LAI "
                  + " / ".join(f"{o[f'ponded_bacc_lai_{lo}-{hi}']:.3f}" for lo, hi in LAI_BINS)
                  + f" | NISAR day, canopy {o['pass_day_bacc_canopy']:.3f} ({time.time() - t:.0f} s)", flush=True)
        summary[name] = res
    (OUT / "nisar_input.json").write_text(json.dumps(summary, indent=2))
    plot(summary)


def plot(summary):
    tests = list(summary)
    variants = [("v1b", "#9ec5f0", "v1b (S2 + S1)"), ("v1n", "#8b5cf6", "v1n (+ NISAR, trained at 12 days)"),
                ("v1n6", "#5b21b6", "v1n6 (+ NISAR, trained at 6 days)"), ("v1n, no NISAR", "#d9cdfa", "v1n / v1n6 with NISAR missing")]
    panels = [("fc_next_dry_err_days", "Forecast next-dry error (days)"), ("fc_ponded_today_bacc", "Ponded today? (balanced acc.)"),
              ("pass_day_bacc_canopy", "Ponded? on NISAR days, LAI ≥ 2"), ("sp_next_dry_err_days", "Season S&P: next-dry error (days)")]
    y = np.arange(len(tests))
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5.4), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        for j, (v, c, lab) in enumerate(variants):
            get = lambda t: summary[t].get(v, summary[t].get("v1n6, no NISAR") if v == "v1n, no NISAR" else None)  # noqa: E731
            vals = [get(t)[m] if get(t) else np.nan for t in tests]
            ax.barh(y + (j - 1.5) * 0.2, vals, height=0.19, color=c, label=lab)
        if "bacc" in m:
            ax.set_xlim(0.5, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace(", NISAR", ",\nNISAR") for t in tests], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("NISAR L-band as a model input (simulated; every 12 days and wet/dry accuracy 0.8 unless stated)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig22_nisar_input.png", dpi=150)




def fusion_check():
    """Part 3: NISAR through an explicit observation model instead of a learned input. On each NISAR
    acquisition day (>= day 25) under the canopy (LAI >= 2): v1b's prior P(ponded) (end of the previous day),
    the raw NISAR call, and their Bayes fusion with NISAR's accuracy (kp.fuse_pass), against the truth at
    the 06:00 pass; and the same fusion applied to v1n's prior."""
    inp = ps.load_inputs()
    d, d6 = sets(inp), sets6(inp)
    models = {}
    for name, path in (("v1b", OUT / "models" / "kgml_paddy_v1b_pretrained.pt"), ("v1n", V1N)):
        models[name] = kp.PaddyNet()
        models[name].load_state_dict(torch.load(path))
    out = {}
    for name, te, revisit, acc in [(n, d[f"test/{n}"], 12, TESTS[n][2]) for n in TESTS] + [(n, d6[f"test/{n}"], 6, 0.8) for n in TESTS6]:
        X = te["X"]
        res = {k: {"truth": [], "p": []} for k in ("v1b prior", "v1n prior", "v1b + NISAR (Bayes)", "v1n + NISAR (Bayes)")}
        raw_t, raw_p = [], []
        cal = {"field": [], "logit": [], "truth": [], "said": []}   # for the logger-calibrated fusion
        for day in range(5, X.shape[1] - 1, revisit):
            if day < 25:
                continue
            lai = ps.lai_curve(inp["s0"] + day + 0.25, te["tp_abs"], te["g"])
            said = X[:, day, kp.CH["nisar_wet"]] > 0
            seen = said | (X[:, day, kp.CH["nisar_dry"]] > 0)
            ok = seen & (lai >= 2) & np.isfinite(te["h6"][:, day * ps.PER_DAY + 1])
            truth = te["h6"][ok, day * ps.PER_DAY + 1] > 0
            raw_t.append(truth); raw_p.append(said[ok])
            cal["field"].append(np.flatnonzero(ok)); cal["truth"].append(truth); cal["said"].append(said[ok])
            for m in ("v1b", "v1n"):
                Xm = no_nisar(X) if m == "v1b" else X
                _, st = kp.predict_v1(models[m], Xm, te["static"], day, 8)   # data before the pass day
                prior = st["p_wet"][ok, day - 1]
                post = kp.fuse_pass(prior, said[ok], np.full(ok.sum(), acc), np.full(ok.sum(), acc))
                for k, p_ in ((f"{m} prior", prior), (f"{m} + NISAR (Bayes)", post)):
                    res[k]["truth"].append(truth); res[k]["p"].append(p_ > 0.5)
                if m == "v1b":
                    q = np.clip(prior, 1e-4, 1 - 1e-4)
                    cal["logit"].append(np.log(q / (1 - q)))
        o = {"NISAR alone": m11.bacc(np.concatenate(raw_t), np.concatenate(raw_p)), "n": int(sum(len(t) for t in raw_t))}
        o.update({k: m11.bacc(np.concatenate(v["truth"]), np.concatenate(v["p"])) for k, v in res.items()})
        # Experiment 14's fix: recalibrate the prior (2-parameter logistic) on 40 logger fields, fuse, and
        # score on the other fields only
        fid, x, y, sd = (np.concatenate(cal[k]) for k in ("field", "logit", "truth", "said"))
        loggers = np.random.default_rng(23).choice(len(X), 40, replace=False)
        fit = np.isin(fid, loggers)
        a, b = 0.0, 1.0
        for _ in range(30):
            z = 1 / (1 + np.exp(-(a + b * x[fit])))
            g = np.array([np.sum(z - y[fit]), np.sum((z - y[fit]) * x[fit])])
            w = z * (1 - z)
            Hs = np.array([[w.sum(), (w * x[fit]).sum()], [(w * x[fit]).sum(), (w * x[fit] ** 2).sum()]]) + 1e-3 * np.eye(2)
            a, b = np.array([a, b]) - np.linalg.solve(Hs, g)
        ev = ~fit
        prior_cal = 1 / (1 + np.exp(-(a + b * x[ev])))
        post_cal = kp.fuse_pass(prior_cal, sd[ev], np.full(ev.sum(), acc), np.full(ev.sum(), acc))
        o["v1b + NISAR (Bayes, prior calibrated on 40 loggers)"] = m11.bacc(y[ev] > 0.5, post_cal > 0.5)
        o["NISAR alone, same fields"] = m11.bacc(y[ev] > 0.5, sd[ev])
        o["calibration slope"] = float(b)
        out[name] = o
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    s = json.loads((OUT / "nisar_input.json").read_text())
    s["_fusion_on_pass_days"] = out
    (OUT / "nisar_input.json").write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot({k: v for k, v in json.loads((OUT / "nisar_input.json").read_text()).items() if not k.startswith("_")})
    elif "--fusion" in sys.argv:
        fusion_check()
    else:
        main()
        fusion_check()
