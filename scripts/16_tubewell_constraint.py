"""Experiment 16: the tubewell-total constraint.

A tubewell's meter (BMDA prepaid cards, or a flow meter on the campaign pumps) records the water
pumped to all the fields it serves. Through the water balance, that constrains the sum of those
fields' water losses, so it can correct a systematic bias in the model's S&P for the command area,
as county yields constrained a fine-scale model in KGML-ag-Carbon. It cannot correct field-to-field
scatter.

KGML-Paddy v1 (pre-trained) has two such biases: fields whose farmers do not log irrigations get S&P
about 60 % too low (the network was only trained on logging farmers), and with every pump rated 25 %
high the logging fields get S&P about 11 % too high.

Set-up per world (Experiment 11 test sets): command areas of 25 fields with random areas (0.1-0.3 ha);
a share s of farmers log (s = 100, 50, 20 %; the others' log channels are removed). Meter:
V = sum(area x true irrigation) x (1 + conveyance loss U(5 %, 15 %)) x meter error (5 %); the correction
assumes 10 % conveyance loss.

Correction: the model's seasonal water use per field is W = ET - rain - S_start + S_end + kappa * SP_sum,
with ET from nominal Kc x ET0, the storage terms from the model's daily level, and SP_sum the sum of
SP(h_t, tau_t) under the model's S&P law and level path. kappa per command area matches sum(area x W)
to the meter, and multiplies the effective S&P of
  all      every field in the command area, or
  nonlog   only the fields without farmer logs (loggers kept as estimated);
and the same with the simulator reference ("_simref"): even with everyone logging, the model-based
accounting in the training world gives kappa of about 0.89, not 1 (the model's level path over-predicts
water use a little). That reference is known from simulation alone, before any real data, so kappa is
divided by it (all) or the loggers' S&P part is scaled by it (nonlog).
Evaluated on the effective S&P (next-dry error, % error, bias) for logging and non-logging fields.

Outputs: results/tubewell_constraint.json, results/figures/fig15_tubewell_constraint.png.
Runtime about 5 minutes.
"""

import json
import pathlib
import sys

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
WORLDS = ("prior", "reality", "reality_systematic")
SHARES = (1.0, 0.5, 0.2)
TW_SIZE = 25
SY_REG = 0.065
ASSUMED_CONVEYANCE = 0.10
CH = kp.CH


def water_terms(te, th_med, st, inp):
    """Per field: non-S&P part of seasonal irrigation (ET - rain - S_start + S_end, mm) and the
    S&P sum (mm) under the model's law and daily level path. Days on which the level sits at the floor
    of the range (soil water used up) carry no losses, as in the simulator; without this the
    accounting over-predicts irrigation by about 8 % even with the true law and level."""
    X = te["X"]
    n, T = X.shape[:2]
    s0 = inp["s0"]
    season = X[:, :, CH["season"]] > 0
    days = np.arange(T)
    t_mid = s0 + days[None, :] + 0.5
    tp = te["tp_abs"][:, None]
    etc = ps.kc_rice(ps.lai_curve(t_mid, tp)) * inp["et0"][s0:s0 + T][None, :]
    rain = np.broadcast_to(inp["rain"][s0:s0 + T][None, :], (n, T))
    h = st["h"]
    sp = ps.sp_rate(*(th_med[:, i:i + 1] for i in range(4)), h, t_mid - tp)
    first = np.argmax(season, 1)
    storage = lambda x: np.where(x > 0, x, SY_REG * x)  # noqa: E731
    s_start = storage(st["pos_mu"][np.arange(n), first])  # the puddling flood (not a metered irrigation)
    s_end = storage(h[:, -2])
    active = season & (h > ps.PIPE_MIN + 5)
    non_sp = (etc * active).sum(1) - (rain * season).sum(1) - s_start + s_end
    return non_sp, (sp * active).sum(1)


def main():
    torch.set_num_threads(4)
    inp = ps.load_inputs()
    et_mid = float(inp["et0"][(pd.Timestamp("2024-02-01") - inp["day0"]).days:
                              (pd.Timestamp("2024-04-15") - inp["day0"]).days].mean())
    d = m11.sets(inp)
    model = kp.PaddyNet()
    model.load_state_dict(torch.load(OUT / "models" / "kgml_paddy_v1_pretrained.pt"))
    rows, kappas = [], {}
    k_ref = None  # set in the training world with everyone logging (WORLDS starts with "prior")
    for w in WORLDS:
        te = d[f"test/{w}"]
        n, T = te["X"].shape[:2]
        rng = np.random.default_rng(7)
        area = rng.uniform(0.1, 0.3, n)
        tw = np.arange(n) // TW_SIZE
        n_tw = tw.max() + 1
        conveyance = rng.uniform(0.05, 0.15, n_tw)
        meter_err = rng.lognormal(0, 0.05, n_tw)
        V = np.bincount(tw, area * te["irr_true_mm"]) * (1 + conveyance) * meter_err  # mm x ha
        for share in SHARES:
            logs = np.random.default_rng(int(share * 100)).random(n) < share
            X = te["X"].copy()
            X[np.ix_(~logs, np.arange(T), [CH["irr_logged"], CH["depth_logged"]])] = 0
            th, st = kp.predict_v1(model, X, te["static"], T, 100)
            th_med = np.median(th, 0)
            sp_samples = ps.sp_effective(*(th[..., i] for i in range(4)), te["tp_abs"][None, :], inp, dt=0.05)
            sp_hat = np.median(sp_samples, 0)
            non_sp, sp_sum = water_terms({**te, "X": X}, th_med, st, inp)
            V_field = V / (1 + ASSUMED_CONVEYANCE)
            fixed_all = np.bincount(tw, area * non_sp)
            k_all = np.clip((V_field - fixed_all) / np.bincount(tw, area * sp_sum), 0.2, 6.0)
            # non-loggers only: logging fields' water use kept at the model's estimate
            fixed_nl = fixed_all + np.bincount(tw, area * sp_sum * logs)
            denom = np.bincount(tw, area * sp_sum * ~logs)
            k_nl = np.where(denom > 0, np.clip((V_field - fixed_nl) / np.maximum(denom, 1e-9), 0.2, 6.0), 1.0)
            if w == "prior" and share == 1.0:
                k_ref = float(np.median(k_all))
            fixed_nl_ref = fixed_all + k_ref * np.bincount(tw, area * sp_sum * logs)
            k_nl_ref = np.where(denom > 0, np.clip((V_field - fixed_nl_ref) / np.maximum(denom, 1e-9), 0.2, 6.0), 1.0)
            kappas[f"{w}|{share}"] = {"all_median": float(np.median(k_all)), "nonlog_median": float(np.median(k_nl)),
                                      "reference": k_ref}
            variants = {"model": sp_hat, "tubewell_all": sp_hat * k_all[tw],
                        "tubewell_nonlog": np.where(logs, sp_hat, sp_hat * k_nl[tw]),
                        "tubewell_all_simref": sp_hat * (k_all / k_ref)[tw],
                        "tubewell_nonlog_simref": np.where(logs, sp_hat, sp_hat * k_nl_ref[tw])}
            for name, pred in variants.items():
                for group, sel in (("loggers", logs), ("non_loggers", ~logs)):
                    if sel.sum() < 20:
                        continue
                    y = te["sp_eff"][sel]
                    m = m9.point_metrics(pred[sel], y, et_mid)
                    rows.append({"world": w, "share_logging": share, "variant": name, "group": group,
                                 "n": int(sel.sum()), "log_bias": float(np.median(np.log(pred[sel] / y))), **m})
            print(f"{w} share {share:.0%}: kappa all {np.median(k_all):.2f} (p10-p90 "
                  f"{np.percentile(k_all, 10):.2f}-{np.percentile(k_all, 90):.2f}), non-loggers {np.median(k_nl):.2f}",
                  flush=True)
    df = pd.DataFrame(rows)
    summary = {"rows": df.round(4).to_dict(orient="records"), "kappa": kappas}
    (OUT / "tubewell_constraint.json").write_text(json.dumps(summary, indent=2))
    print(df.pivot_table(index=["world", "share_logging", "group"], columns="variant",
                         values=["next_dry_err_days", "log_bias"]).round(2).to_string())
    plot(summary)


def plot(summary):
    df = pd.DataFrame(summary["rows"])
    colors = {"model": ("#8a8983", "Model alone"),
              "tubewell_all_simref": ("#2a78d6", "Meter correction, all fields (simulator-referenced)"),
              "tubewell_nonlog_simref": ("#eb6834", "Meter correction, non-loggers only (simulator-referenced)")}
    titles = {"prior": "Training world", "reality": "Random mismatch", "reality_systematic": "Random + systematic"}
    fig, axes = plt.subplots(2, len(WORLDS), figsize=(12, 6.4), sharey="row")
    for c, w in enumerate(WORLDS):
        for r, group in enumerate(("loggers", "non_loggers")):
            ax = axes[r, c]
            for v, (col, lab) in colors.items():
                g = df[(df.world == w) & (df.group == group) & (df.variant == v)].sort_values("share_logging", ascending=False)
                if g.empty:
                    continue
                ax.plot([f"{int(s * 100)} %" for s in g.share_logging], g.next_dry_err_days, marker="o", ms=5, lw=2,
                        color=col, label=lab)
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(titles[w], fontsize=10)
            if c == 0:
                ax.set_ylabel(("Fields that log" if group == "loggers" else "Fields that do not log") +
                              "\nS&P: next-dry error (days)")
            if r == 1:
                ax.set_xlabel("Share of farmers who log")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Tubewell meter totals as a constraint on the S&P of the fields they serve (25 fields per tubewell)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / "fig15_tubewell_constraint.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "tubewell_constraint.json").read_text()))
    else:
        main()
