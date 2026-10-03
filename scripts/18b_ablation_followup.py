"""Experiment 18b: the two settings the pre-trained ablation (Experiment 18) does not cover.

1. Fields whose farmers do not log. Experiment 18's test fields all keep their farmer log. Here half of
   them lose it (as in Experiment 17), and the satellite (S) and structure (F) effects on season S&P
   and on the forecast are measured separately for fields with and without a log. Pre-trained models
   (P1), with the physics loss (L1).
2. Fine-tuning on loggers. Experiments 9-10 found the physical structure's advantage when the model is
   fine-tuned through the S&P law under mismatch. Here S1-F1-L1 (v1b), S1-F0-L1 (black-box S&P head)
   and S1-F1-L0 (no physics loss, also not in fine-tuning) are fine-tuned on 80 logger fields
   (Experiment 12's recipe; the black-box head is fine-tuned on each logger's own effective S&P) in
   the two mismatched worlds, 3 logger draws each.

Outputs: results/ablation_followup.json. Runtime about 30 minutes. Needs the weights from script 18.
"""

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
m11 = __import__("11_kgml_paddy_v1_state")
m17 = __import__("17_train_with_nonloggers")
m18 = __import__("18_ablation")
from awdrt import kgml_paddy as kp  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402

OUT = ROOT / "results"
N_FT = 80
FT_SEEDS = {"reality": (61, 62, 63), "reality_systematic": (64, 65, 66)}
FT_VARIANTS = ((1, 1, 1), (1, 0, 1), (1, 1, 0))  # (S, F, L)


def load(S, F, L):
    m = kp.PaddyNet(structure=bool(F))
    m.load_state_dict(torch.load(m18.WDIR / f"{m18.name(S, 1, F, L)}.pt"))
    return m


def logger_set(w, inp, seed, n):
    lg = ps.simulate(ps.WORLDS[w], 2 * n, seed, inp)
    lg["H_log"] = ps.end_of_day_depth(lg["logger"])
    lg.update(zip(("h", "tau", "r"), ps.recession_rates(lg["logger"], lg["tp_abs"], inp)))
    lg["sp_log"] = kp.logger_sp_eff(lg, inp)
    keep = np.flatnonzero(np.isfinite(lg["r"]).sum(1) >= 8)[:n]
    k = len(lg["theta"])
    return {key: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (k,) else v) for key, v in lg.items()
            if key not in ("logger", "h6")}


def metrics(model, te, X, inp, et_mid, groups):
    sp, _, _ = m17.season_metrics(model, te, X, inp, et_mid)
    fc = m17.forecast_group(model, te, X, inp, groups)
    return {g: {**m17.group_sp(sp, te, sel, et_mid), **{f"fc_{k}": v for k, v in fc[g].items()}}
            for g, sel in groups.items()}


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    summary = {"nonloggers": {}, "fine_tune": {}}

    # 1. fields without a farmer log
    for w in m18.WORLDS:
        te = d[f"test/{w}"]
        logs = np.random.default_rng(50).random(len(te["X"])) < m17.SHARE_LOGGING
        groups = {"loggers": logs, "non_loggers": ~logs}
        res = {}
        for S in (0, 1):
            for F in (0, 1):
                X = kp.drop_logs(te["X"], ~logs)
                X = X if S else kp.drop_satellite(X)
                res[m18.name(S, 1, F, 1)] = metrics(load(S, F, 1), te, X, inp, et_mid, groups)
        summary["nonloggers"][w] = res
        for v, r in res.items():
            for g in groups:
                o = r[g]
                print(f"{w} {v} {g:11s}: S&P bias {o['sp_log_bias']:+.2f}, err {o['sp_next_dry_err_days']:.2f} d, "
                      f"rho {o['sp_spearman']:.2f} | forecast {o['fc_next_dry_err_days']:.2f} d", flush=True)

    # 2. fine-tuning on loggers
    for w, seeds in FT_SEEDS.items():
        te = d[f"test/{w}"]
        allf = {"all": np.ones(len(te["X"]), bool)}
        rows = []
        for r_, seed in enumerate(seeds):
            lg = logger_set(w, inp, seed, N_FT)
            for S, F, L in FT_VARIANTS:
                t = time.time()
                v = m18.name(S, 1, F, L)
                base = metrics(load(S, F, L), te, te["X"], inp, et_mid, allf)["all"] if r_ == 0 else None
                ft = kp.fine_tune_v1(load(S, F, L), lg, d["train"], phys_w=0.01 if L else 0.0, seed=r_)
                o = metrics(ft, te, te["X"], inp, et_mid, allf)["all"]
                rows.append({"variant": v, "draw": r_, **{f"ft_{k}": x for k, x in o.items()}})
                if base is not None:
                    summary["fine_tune"].setdefault(w, {})[f"{v}|pretrained"] = base
                print(f"{w} draw {r_} {v}: fine-tuned S&P err {o['sp_next_dry_err_days']:.2f} d "
                      f"(bias {o['sp_log_bias']:+.2f}), forecast {o['fc_next_dry_err_days']:.2f} d "
                      f"({time.time() - t:.0f} s)", flush=True)
        df = pd.DataFrame(rows)
        agg = df.drop(columns="draw").groupby("variant").agg(["mean", "std"])
        for v in agg.index:
            summary["fine_tune"][w][f"{v}|fine_tuned"] = {f"{k}_{s}": float(agg.loc[v, (k, s)])
                                                           for k, s in agg.columns}
        print(agg.round(3).to_string(), flush=True)
    (OUT / "ablation_followup.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
