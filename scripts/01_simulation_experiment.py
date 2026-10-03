"""Experiment 1 (physics feasibility): can Sentinel-2 see the background under rice?

E1  Background separability vs LAI. A classifier trained on PROSAIL simulations
    predicts ponded / saturated / drying background. It is compared against the
    two rules used in practice: Xiao et al. (2005) "LSWI + 0.05 >= EVI" flood
    signal and MNDWI > 0.
E2  LAI retrieval bias. A soil-background retriever (the standard hybrid
    PROSAIL+ML set-up, as in the SNAP biophysical processor) versus a
    background-aware retriever, both tested on ponded fields.

Outputs go to results/ (metrics JSON, figures, and the trained models reused by
scripts/02_real_s2_boro_jamalpur.py).
"""

import json
import pathlib
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import backgrounds, rtm  # noqa: E402
from awdrt.s2 import FEATURE_BANDS, WL, convolve, indices  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(exist_ok=True)

N_TRAIN, N_TEST = 30000, 15000
LAI_BINS = np.arange(0, 6.01, 0.5)


def features(df):
    X = df[FEATURE_BANDS].to_numpy()
    ix = indices(X)
    return np.column_stack([X, ix["ndvi"], ix["lswi"], ix["mndwi"]])


def by_lai(df, fn):
    out = []
    for lo, hi in zip(LAI_BINS[:-1], LAI_BINS[1:]):
        m = (df.lai >= lo) & (df.lai < hi)
        out.append(fn(df[m]))
    return out


def main():
    t0 = time.time()
    train = rtm.simulate(N_TRAIN, np.random.default_rng(1))
    test = rtm.simulate(N_TEST, np.random.default_rng(2))
    print(f"simulated {N_TRAIN + N_TEST} spectra in {time.time() - t0:.0f}s")

    Xtr, Xte = features(train), features(test)
    ponded_tr, ponded_te = (train.background == "ponded").to_numpy(), (test.background == "ponded").to_numpy()

    # ── E1: background classification ────────────────────────────────────────
    clf3 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    clf3.fit(Xtr, train.background)
    clf2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    clf2.fit(Xtr, ponded_tr)
    test["pred3"] = clf3.predict(Xte)
    test["p_ponded"] = clf2.predict_proba(Xte)[:, 1]
    ix = indices(test[FEATURE_BANDS].to_numpy())
    test["xiao"] = ix["lswi"] + 0.05 >= ix["evi"]
    test["mndwi_rule"] = ix["mndwi"] > 0
    test["is_ponded"] = ponded_te

    def bacc(col):
        return lambda d: balanced_accuracy_score(d.is_ponded, d[col]) if d.is_ponded.nunique() == 2 else np.nan

    test["rt_ponded"] = test.p_ponded > 0.5
    e1 = {
        "lai_bin_lo": LAI_BINS[:-1].tolist(),
        "rt_binary_bacc": by_lai(test, bacc("rt_ponded")),
        "xiao_bacc": by_lai(test, bacc("xiao")),
        "mndwi_bacc": by_lai(test, bacc("mndwi_rule")),
        "rt_3class_acc": by_lai(test, lambda d: float((d.pred3 == d.background).mean())),
        # How often each rule flags a ponded field as flooded (recall).
        "xiao_recall_ponded": by_lai(test, lambda d: float(d[d.is_ponded].xiao.mean())),
        "rt_recall_ponded": by_lai(test, lambda d: float(d[d.is_ponded].rt_ponded.mean())),
    }

    def limit(curve, thr=0.8):
        """Highest LAI bin (upper edge) up to which accuracy stays >= thr."""
        ok = np.array(curve) >= thr
        return float(LAI_BINS[1:][np.argmin(ok)]) if not ok.all() else float(LAI_BINS[-1])

    e1["detect_limit_lai_rt"] = limit(e1["rt_binary_bacc"])
    e1["detect_limit_lai_xiao"] = limit(e1["xiao_bacc"])
    e1["overall_rt_bacc_lai_lt_2"] = float(balanced_accuracy_score(
        test.is_ponded[test.lai < 2], test.rt_ponded[test.lai < 2]))
    e1["overall_xiao_bacc_lai_lt_2"] = float(balanced_accuracy_score(
        test.is_ponded[test.lai < 2], test.xiao[test.lai < 2]))

    # Permutation-free band importance: accuracy drop without the SWIR bands.
    no_swir = [i for i, b in enumerate(FEATURE_BANDS) if b not in ("B11", "B12")]
    clf_ns = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    clf_ns.fit(Xtr[:, no_swir], ponded_tr)
    e1["rt_bacc_lai_lt_2_without_swir"] = float(balanced_accuracy_score(
        ponded_te[test.lai < 2], clf_ns.predict(Xte[:, no_swir])[test.lai < 2]))

    # ── E2: LAI retrieval with and without background awareness ─────────────
    soil_only = train.background != "ponded"
    reg_soil = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=0)
    reg_soil.fit(Xtr[soil_only], train.lai[soil_only])
    reg_aware = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=0)
    reg_aware.fit(Xtr, train.lai)
    reg_cwc = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=0)
    reg_cwc.fit(Xtr, train.cwc)
    test["lai_soil"] = reg_soil.predict(Xte)
    test["lai_aware"] = reg_aware.predict(Xte)
    test["cwc_aware"] = reg_cwc.predict(Xte)

    pond = test[test.is_ponded]
    e2 = {
        "lai_bin_lo": LAI_BINS[:-1].tolist(),
        "ponded_bias_soil_model": by_lai(pond, lambda d: float((d.lai_soil - d.lai).mean())),
        "ponded_bias_aware_model": by_lai(pond, lambda d: float((d.lai_aware - d.lai).mean())),
        "ponded_rmse_soil_model": by_lai(pond, lambda d: float(np.sqrt(((d.lai_soil - d.lai) ** 2).mean()))),
        "ponded_rmse_aware_model": by_lai(pond, lambda d: float(np.sqrt(((d.lai_aware - d.lai) ** 2).mean()))),
    }
    early = pond[pond.lai < 2]
    e2["ponded_lai_lt_2_rmse_soil"] = float(np.sqrt(((early.lai_soil - early.lai) ** 2).mean()))
    e2["ponded_lai_lt_2_rmse_aware"] = float(np.sqrt(((early.lai_aware - early.lai) ** 2).mean()))
    e2["ponded_lai_lt_2_bias_soil"] = float((early.lai_soil - early.lai).mean())
    e2["ponded_lai_lt_2_bias_aware"] = float((early.lai_aware - early.lai).mean())
    e2["all_rmse_aware"] = float(np.sqrt(((test.lai_aware - test.lai) ** 2).mean()))
    e2["cwc_r2_aware"] = float(1 - ((test.cwc_aware - test.cwc) ** 2).sum() / ((test.cwc - test.cwc.mean()) ** 2).sum())

    metrics = {"n_train": N_TRAIN, "n_test": N_TEST, "E1_background": e1, "E2_lai": e2}
    (OUT / "simulation_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    with open(OUT / "models" / "background_aware_models.pkl", "wb") as f:
        pickle.dump({"clf_ponded": clf2, "clf_3class": clf3, "reg_lai": reg_aware,
                     "reg_lai_soil_only": reg_soil, "reg_cwc": reg_cwc,
                     "features": "FEATURE_BANDS + ndvi, lswi, mndwi"}, f)
    # Keep the simulated envelope for the sim-to-real check on real pixels.
    test[FEATURE_BANDS + ["lai", "background"]].to_csv(OUT / "sim_test_spectra.csv.gz", index=False)

    plot(test, e1, e2)
    print(json.dumps({k: v for k, v in e1.items() if not isinstance(v, list)}, indent=2))
    print(json.dumps({k: v for k, v in e2.items() if not isinstance(v, list)}, indent=2))


def plot(test, e1, e2):
    mid = LAI_BINS[:-1] + 0.25

    # Fig 1: what the canopy does to the background contrast.
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    import prosail
    for ax, lai in zip(axes, [0.3, 1.5, 4.0]):
        for cls, col in zip(backgrounds.CLASSES, ["#1f77b4", "#8c564b", "#e6a23c"]):
            bg, _ = backgrounds.sample_background(cls, rng)
            if cls == "drying":
                bg = backgrounds.soil(0.6, 1.0)
            r = prosail.run_prosail(1.5, 40, 8, 0.0, 0.015, 0.005, lai, 62, 0.1, 40, 5, 90,
                                    typelidf=2, rsoil0=bg, factor="SDR")
            ax.plot(WL, r, color=col, lw=1.2, label=cls)
            ax.scatter([rtm_c for rtm_c in (864.7, 1613.7, 2202.4)],
                       convolve(r)[0][[6, 7, 8]], color=col, s=18, zorder=3)
        ax.set_title(f"rice canopy, LAI = {lai}")
        ax.set_xlabel("wavelength (nm)")
        ax.axvspan(1560, 1660, color="0.9", zorder=0)
        ax.axvspan(2110, 2290, color="0.9", zorder=0)
    axes[0].set_ylabel("canopy reflectance")
    axes[0].legend(frameon=False)
    fig.suptitle("PROSAIL: the background signal fades as the rice canopy closes (grey = S2 B11/B12)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_spectra_by_background.png", dpi=150)
    plt.close(fig)

    # Fig 2: detection accuracy vs LAI.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(mid, e1["rt_binary_bacc"], "o-", label="RT-trained classifier (this work)")
    ax.plot(mid, e1["xiao_bacc"], "s--", label="Xiao 2005 rule (LSWI+0.05 ≥ EVI)")
    ax.plot(mid, e1["mndwi_bacc"], "^:", label="MNDWI > 0")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.axhline(0.8, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("LAI")
    ax.set_ylabel("balanced accuracy: standing water yes/no")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Detecting standing water under rice (simulated S2, with noise)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_water_detection_vs_lai.png", dpi=150)
    plt.close(fig)

    # Fig 3: LAI bias when the background is water.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(mid, e2["ponded_bias_soil_model"], "o-", label="soil-background retriever (SNAP-style)")
    ax.plot(mid, e2["ponded_bias_aware_model"], "s-", label="background-aware retriever")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("true LAI")
    ax.set_ylabel("LAI bias on ponded fields")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Ignoring the water background biases early-season LAI")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_lai_bias_ponded.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
