"""Experiment 27: realistic rice growth (Boro phenology in thermal time) and what it does to sensing and the model.

Experiments 1-26 grew the crop on a straight line: LAI 0.5 at transplanting, +0.0375 a day. Transplanted Boro
rice grows slowly in the cold of January and fast through tillering in February and March, closing its canopy
(LAI about 5.5) around booting; plants grow from about 20 to 95 cm and tiller from about 3 to 22 per hill.
Simulator v2 now has that growth (paddy_sim PHENOLOGY = "boro": thermal time above 10 C from ERA5), with
NISAR's accuracy falling above 70 cm plant height. This experiment:

  1. Satellite skill by growth stage: how often Sentinel-2 and Sentinel-1 wet/dry calls are wrong at each
     stage, with the old and the new growth (the same test worlds otherwise).
  2. Pre-trains KGML-Paddy v1d exactly as v1b (Experiment 17: 20,000 fields, 30 % without a farmer log), but
     on the simulator with Boro growth.
  3. Compares v1b (trained on the straight-line crop) and v1d on Boro-growth test worlds (training world,
     random mismatch, random + systematic; 2,000 fields each): season S&P and forecasts from days 45, 55, 65.

Outputs: results/boro_phenology.json, results/models/kgml_paddy_v1d_pretrained.pt,
results/figures/fig26_boro_phenology.png. Runtime about 20 minutes (`--plot-only` redraws the figure).
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
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "kgml_v1_sets_boro.pkl"
FIG = OUT / "figures" / "fig26_boro_phenology.png"
TESTS = m11.TESTS
DAYS = m11.FORECAST_DAYS
P_NOLOG = 0.3
DAT_BINS = [(0, 20), (20, 40), (40, 55), (55, 70), (70, 120)]


def sets_boro(inp):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    out = {"train": ps.simulate(ps.WORLDS["prior"], 20_000, 1, inp), "val": ps.simulate(ps.WORLDS["prior"], 2_000, 2, inp)}
    for name, (w, seed) in TESTS.items():
        out[f"test/{name}"] = ps.simulate(ps.WORLDS[w], 2_000, seed, inp)
    for k, d in out.items():
        d["H"] = ps.end_of_day_depth(d["h6"])
        drop = ("logger", "h6") if k in ("train", "val") else ("logger",)
        out[k] = {kk: v for kk, v in d.items() if kk not in drop}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def sensing_by_stage(d, inp):
    """Wrong-call rate of Sentinel-2 (12:00 reading) and Sentinel-1 (18:00) by days after transplanting."""
    X, h6, tp = d["X"], d["h6"], d["tp_abs"] - inp["s0"]
    out = {}
    for s, q in (("s2", 2), ("s1", 3)):
        f, day = np.nonzero((X[:, :, kp.CH[f"{s}_wet"]] + X[:, :, kp.CH[f"{s}_dry"]]) > 0)
        said = X[f, day, kp.CH[f"{s}_wet"]] > 0
        truth = h6[f, day * ps.PER_DAY + q] > 0
        dat = day + 0.5 - tp[f]
        out[s] = {}
        for a, b in DAT_BINS:
            m = (dat >= a) & (dat < b)
            out[s][f"{a}-{b}"] = {"wrong": float((said[m] != truth[m]).mean()) if m.any() else None,
                                  "wet_missed": float((~said[m & truth]).mean()) if (m & truth).any() else None, "n": int(m.sum())}
    return out


def evaluate(model, d, inp, et_mid):
    th, st = kp.predict_v1(model, d["X"], d["static"], d["X"].shape[1], 100)
    sp = np.median(ps.sp_effective(*(th[..., i] for i in range(4)), d["tp_abs"][None, :], inp, dt=0.05), 0)
    m = m9.point_metrics(sp, d["sp_eff"], et_mid)
    out = {"season_sp_next_dry_err": m["next_dry_err_days"], "season_sp_spearman": float(spearmanr(sp, d["sp_eff"]).correlation)}
    fc = [m11.forecast_metrics(model, d, inp, day) for day in DAYS]
    out["ponded_today_bacc"] = float(np.mean([f["ponded_today_bacc_causal"] for f in fc]))
    out["forecast_err"] = float(np.mean([f["kgml"]["median_abs_err_days"] for f in fc]))
    out["forecast_cov90"] = float(np.mean([f["kgml"]["ci90_coverage"] for f in fc]))
    out["regional_err"] = float(np.mean([f["regional_sp_kgml_depth"]["median_abs_err_days"] for f in fc]))
    out["et_only_err"] = float(np.mean([f["et_only_kgml_depth"]["median_abs_err_days"] for f in fc]))
    return out


def run():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:(pd.Timestamp("2024-03-31") - inp["day0"]).days].mean())
    res = {"sensing": {"linear": {}, "boro": {}}, "models": {}}
    lin = m11.sets(inp)                               # Experiment 11's straight-line test worlds (cached)
    for name in TESTS:
        res["sensing"]["linear"][name] = sensing_by_stage(lin[f"test/{name}"], inp)
    del lin
    ps.set_phenology("boro")
    t = time.time()
    d = sets_boro(inp)
    print(f"Boro sets ready in {time.time() - t:.0f} s", flush=True)
    for name in TESTS:
        res["sensing"]["boro"][name] = sensing_by_stage(d[f"test/{name}"], inp)
    path = OUT / "models" / "kgml_paddy_v1d_pretrained.pt"
    if path.exists():
        v1d = kp.PaddyNet()
        v1d.load_state_dict(torch.load(path))
    else:
        t = time.time()
        v1d = kp.pretrain_v1(d["train"], d["val"], epochs=30, p_nolog=P_NOLOG, log=lambda s: print(s, flush=True))
        torch.save(v1d.state_dict(), path)
        print(f"pre-trained v1d in {time.time() - t:.0f} s", flush=True)
    v1b = kp.PaddyNet()
    v1b.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1b_pretrained.pt"))
    for mname, model in (("v1b", v1b), ("v1d", v1d)):
        res["models"][mname] = {}
        for name in TESTS:
            res["models"][mname][name] = evaluate(model, d[f"test/{name}"], inp, et_mid)
            print(mname, name, {k: round(v, 3) for k, v in res["models"][mname][name].items()}, flush=True)
    (OUT / "boro_phenology.json").write_text(json.dumps(res, indent=1))
    return res


def plot(res):
    ps.set_phenology("boro")
    inp = ps.load_inputs()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    tp = (pd.Timestamp("2024-01-25") - inp["day0"]).days
    dat = np.arange(0, 100)
    ps.set_phenology("linear")
    lai_lin = ps.lai_curve(tp + dat, tp)
    ps.set_phenology("boro")
    c = ps.crop_state(tp + dat, tp)
    ax[0].plot(dat, lai_lin, color="#807e78", ls="--", label="LAI, straight line (Exp. 1-26)")
    ax[0].plot(dat, c["lai"], color="#2a78d6", lw=2, label="LAI, Boro thermal time")
    a2 = ax[0].twinx()
    a2.plot(dat, c["height_cm"], color="#1baf7a", label="height (cm)")
    a2.plot(dat, c["tillers"] * 4, color="#eb6834", label="tillers per hill (x4)")
    a2.set_ylabel("cm, tillers x 4")
    ax[0].set_xlabel("days after transplanting (25 Jan)")
    ax[0].set_ylabel("LAI")
    ax[0].set_title("Crop growth")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper left")
    bins = [f"{a}-{b}" for a, b in DAT_BINS]
    x = np.arange(len(bins))
    for j, (s, lab) in enumerate((("s2", "Sentinel-2"), ("s1", "Sentinel-1"))):
        for k, (ph, col) in enumerate((("linear", "#807e78"), ("boro", "#2a78d6"))):
            v = [res["sensing"][ph]["prior"][s][b]["wet_missed"] for b in bins]
            ax[1 + j].bar(x + (k - 0.5) * 0.38, [np.nan if y is None else y for y in v], 0.36, color=col, label=ph)
        ax[1 + j].set_xticks(x, bins)
        ax[1 + j].set_xlabel("days after transplanting")
        ax[1 + j].set_ylabel("ponded fields called dry")
        ax[1 + j].set_title(f"{lab}: water missed under the canopy")
        ax[1 + j].legend(fontsize=8, frameon=False)
        ax[1 + j].set_ylim(0, 1)
    fig.suptitle("Experiment 27: Boro rice growth in thermal time, and the satellites (training world)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def main():
    res = json.loads((OUT / "boro_phenology.json").read_text()) if "--plot-only" in sys.argv else run()
    plot(res)


if __name__ == "__main__":
    main()
