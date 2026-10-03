"""Advice against practice: the same fields and weather, irrigated two ways.

One simulated village (paddy_sim.simulate) is re-run from transplanting under different irrigation policies:

  practice    every farmer irrigates by their own rule (simulator v2: conventional, AWD or topping up)
  model       farmers follow KGML-Paddy's advice. Each evening the model sees what this arm's fields have
              shown so far (satellite passes, the farmer's log, "water gone" reports, weather) and forecasts
              when each field's water will be gone. When that is within a day, the farmer is told to irrigate
              `wait` days after it is gone, and does so that morning
  regional    the same advice from the model's state, but with one regional S&P law for every field
  et_only     the same advice with no S&P at all (evapotranspiration only, as IRAS-like tools)
  perfect     the advice from the true state and the true S&P (the ceiling for any forecast)

Optionally the app asks a following farmer "how much water is in the field now?" when the answer could change the
advice (Experiment 13); the answer updates the model's samples before the advice is given.

Everything that does not depend on the policy is drawn once and shared by every arm: each field's physics, the
puddling flood, the depth of a field's k-th irrigation, cloud and the satellite detectors' noise, which
irrigations the farmer logs, and the pump rating error. So the arms differ only by when water is applied, and
what follows from it (the water level, what satellites see, what the model believes).

The physics is simulator v2's (paddy_sim.simulate), integrated the same way (DT = 0.05 d); the practice arm
reproduces the simulator's own season statistically, not draw for draw.
"""

import numpy as np

from . import kgml_paddy as kp
from . import paddy_sim as ps

POLICIES = ("practice", "model", "regional", "et_only", "perfect")
MAX_DAYS = 15.0
STRESS_MM = -150.0          # below the AWD "safe" threshold (15 cm below the surface): a yield-risk proxy
MAX_IRR = 80                # irrigations per field drawn in advance
ASK_P_CORRECT, ASK_CAT_CORRECT = 0.95, 0.8   # the farmer's answer: water or not, and the depth category


def _shared_draws(sim, cfg, inp, seed):
    """Random numbers that every arm shares, so the arms differ only by their policy."""
    n = len(sim["tp"])
    T = inp["s_end"] - inp["s0"]
    rng = np.random.default_rng(seed + 7919)
    fm = sim["farmer"]
    return {
        "puddle": rng.lognormal(np.log(fm["h0_med"]), 0.2),
        "depth": rng.lognormal(np.log(fm["h0_med"])[:, None], 0.2, (n, MAX_IRR)),
        "delay": rng.random((n, MAX_IRR)),
        "log": rng.random((n, MAX_IRR)),
        "advice_hour": rng.uniform(0.2, 0.4, (n, MAX_IRR)),
        "gone": rng.random((n, T)),
        "s2_valid": rng.random((n, T)),
        "s2": rng.random((n, T)), "s1": rng.random((n, T)), "nisar": rng.random((n, T)),
        "lai": rng.standard_normal((n, T)),
        "follow": rng.random(n),
        "ask_wet": rng.random((n, T)), "ask_cat": rng.random((n, T)),
    }


def _detect(u, truth, lai, rec_bins, spec_bins, lo):
    b = np.clip(np.searchsorted(lo, lai, side="right") - 1, 0, len(lo) - 1)
    rec, spec = np.clip(rec_bins[b], 0.5, 1), np.clip(spec_bins[b], 0.5, 1)
    return np.where(truth, u < rec, u > spec)


def regional_theta(n):
    """One S&P law for every field: the pre-training prior's medians (simulator v2's training world)."""
    c0 = ps.WORLDS["prior"]
    a = c0.a_median
    return np.broadcast_to(np.array([a, a / np.mean(c0.Lp) + c0.s_lat_max / 2, c0.c_max / 2, c0.w_max / 2]), (n, 4)).copy()


def run_arm(sim, cfg, inp, policy, seed=0, wait=1, follow=1.0, model=None, n_samples=24, draws=None, risk=0.5, lead=1.0,
            no_log=None, ask=False):
    """One season under a policy. `follow` is the share of farmers who follow the advice (the others keep their
    own practice); `wait` the days after the water is gone that the advice says to irrigate; `risk` the chance
    of the water being gone by tomorrow at which the advice is given (0.5: the median forecast, as the app's
    alert; lower is more cautious); `lead` how many days ahead the irrigation is fixed (1: the "irrigate by
    tomorrow" alert; more: the farmer books the pump that far ahead and does not look again); `no_log` marks
    farmers who never use the app (no irrigation or "water gone" entries, in every arm); `ask` lets the app ask
    a following farmer how much water is in the field when the answer could change the advice. Returns daily
    arrays and the decision log."""
    assert policy in POLICIES
    draws = draws or _shared_draws(sim, cfg, inp, seed)
    n = len(sim["tp"])
    s0, T = inp["s0"], inp["s_end"] - inp["s0"]
    et0_d, rain_d = inp["et0"], inp["rain"]
    a, b, c, w = sim["theta"].T
    k_lat, Sy, et_f, g, tp = sim["k_lat"], sim["Sy"], sim["et_f"], sim["g"], sim["tp_abs"]
    fm = sim["farmer"]
    nb = np.minimum(np.arange(n) ^ 1, n - 1)          # an odd last field is its own neighbour (no exchange)
    advised = (draws["follow"] < follow) if policy != "practice" else np.zeros(n, bool)
    no_log = np.zeros(n, bool) if no_log is None else np.asarray(no_log, bool)

    days = np.arange(T)
    season = days[None, :] + s0 + 1 >= tp[:, None]
    lai_d = ps.lai_curve(days[None, :] + s0 + 0.5, tp[:, None], g[:, None])
    ch = kp.CH
    X = np.zeros((n, T, len(ps.CHANNELS)), np.float32)
    X[:, :, ch["season"]] = season
    X[:, :, ch["days_since_tp"]] = np.clip(days[None, :] + s0 - tp[:, None], 0, None) / 100 * season
    X[:, :, ch["et0"]] = et0_d[s0:s0 + T][None, :] * et_f[:, None] / 10
    X[:, :, ch["rain"]] = rain_d[s0:s0 + T][None, :] / 20
    X[:, :, ch["lai_obs"]] = np.clip(lai_d + cfg.lai_noise * draws["lai"], 0, 6) / 5 * season
    s2_days = dict(zip(inp["s2_days"], inp["s2_valid"]))
    rec2, spec2 = inp["rec"] + cfg.s2_rec_shift, inp["spec"] + cfg.s2_spec_shift
    lo1 = np.array([bb[0] for bb in inp["s1_bins"]])
    acc1 = np.array([bb[1] for bb in inp["s1_bins"]]) + cfg.s1_acc_shift
    theta_reg = regional_theta(n)

    S = np.zeros(n)
    started = np.zeros(n, bool)
    next_irr = np.full(n, np.inf)
    was_ponded = np.zeros(n, bool)
    n_irr = np.zeros(n, int)
    h6 = np.full((n, T * ps.PER_DAY), np.nan)
    irr_mm = np.zeros((n, T))
    gone_next = np.zeros(n, bool)
    advice_day = np.full((n, T), np.nan)       # evening d: the day the advice said to irrigate
    asked = np.zeros((n, T), bool)             # evening d: the farmer was asked about the water now
    sub = int(round(1 / ps.DT))
    rec_every = int(round(1 / (ps.PER_DAY * ps.DT)))
    rng_state = np.random.default_rng(seed + 31)
    for d in range(T):
        X[:, d, ch["gone"]] = gone_next & season[:, d] & ~no_log
        gone_next[:] = False
        for k in range(sub):
            t = s0 + d + k * ps.DT
            start_now = ~started & (t >= tp)
            if start_now.any():
                S[start_now] = draws["puddle"][start_now]
                started |= start_now
                was_ponded |= start_now
            irr = started & (t >= next_irr)
            if irr.any():
                f = np.flatnonzero(irr)
                j = np.minimum(n_irr[f], MAX_IRR - 1)
                dep = draws["depth"][f, j]
                S[f] += dep
                irr_mm[f, d] += dep
                logged = (draws["log"][f, j] < fm["p_log"][f]) & ~no_log[f]
                X[f[logged], d, ch["irr_logged"]] = 1.0
                X[f[logged], d, ch["depth_logged"]] += dep[logged] * sim["pump_bias"][f[logged]] / 100
                n_irr[f] += 1
                next_irr[f] = np.inf
            h = np.where(S > 0, S, S / Sy)
            tau = t - tp
            lai = ps.lai_curve(t, tp, g)
            etc = ps.kc_rice(lai) * et0_d[s0 + d] * et_f
            base = a * (1 + w * np.clip(tau, 0, None) / 80) * np.exp(-c * np.clip(tau, 0, None))
            sp = np.where(S > 0, base + b * h, base)
            hp = np.clip(h, 0, None)
            dS = (rain_d[s0 + d] - etc - sp - k_lat * (hp - hp[nb])) * ps.DT
            S = np.where(started, np.maximum(S + dS, ps.PIPE_MIN * Sy), 0.0)
            h = np.where(S > 0, S, S / Sy)
            ended = started & was_ponded & (h <= 0)
            gone_next |= ended & fm["reporter"] & (draws["gone"][:, d] < 0.5)
            was_ponded = started & (h > 0)
            own = started & np.isinf(next_irr) & ~advised & (h < fm["thr"])
            if own.any():
                f = np.flatnonzero(own)
                j = np.minimum(n_irr[f], MAX_IRR - 1)
                next_irr[f] = t + fm["dlo"][f] + draws["delay"][f, j] * (fm["dhi"][f] - fm["dlo"][f])
            if k % rec_every == 0:
                h6[:, d * ps.PER_DAY + k // rec_every] = np.where(started, h, np.nan)

        # the day's satellite passes (truth at the pass time, as the simulator)
        wet = lambda q: np.nan_to_num(h6[:, d * ps.PER_DAY + q], nan=-1.0) > 0  # noqa: E731
        if d in s2_days:
            ok = season[:, d] & (draws["s2_valid"][:, d] < s2_days[d])
            o = _detect(draws["s2"][:, d], wet(2), lai_d[:, d], rec2, spec2, inp["lai_lo"])
            X[:, d, ch["s2_wet"]], X[:, d, ch["s2_dry"]] = ok & o, ok & ~o
        if d >= 2 and (d - 2) % 6 == 0:
            o = _detect(draws["s1"][:, d], wet(3), lai_d[:, d], acc1, acc1, lo1)
            X[:, d, ch["s1_wet"]], X[:, d, ch["s1_dry"]] = season[:, d] & o, season[:, d] & ~o
        if cfg.nisar and d >= 5 and (d - 5) % cfg.nisar_revisit == 0:
            tr = wet(1)
            acc = ps.nisar_accuracy(cfg.nisar_acc, ps.crop_state(s0 + d + 0.25, tp, g)["height_cm"])
            o = np.where(tr, draws["nisar"][:, d] < acc, draws["nisar"][:, d] > acc)
            X[:, d, ch["nisar_wet"]], X[:, d, ch["nisar_dry"]] = season[:, d] & o, season[:, d] & ~o

        # the evening's advice for followers without an irrigation already planned
        need = advised & started & np.isinf(next_irr) & season[:, d]
        if not need.any() or d == T - 1:
            continue
        f = np.flatnonzero(need)
        t0 = s0 + d + 1.0
        h_end = S[f] / np.where(S[f] > 0, 1, Sy[f])
        if policy == "perfect":
            D = kp.forecast_dry(sim["theta"][f], h_end, t0, tp[f], inp, et_f=et_f[f], g=g[f], max_days=MAX_DAYS, dt=0.1)
            dry_now = h_end <= 0
        else:
            th, st = kp.predict_v1(model, X[f], sim["static"][f], d + 1, n_samples=n_samples, seed=seed + d)
            hs = kp.sample_depth(st, d, n_samples, rng_state)
            if policy == "regional":
                th = np.broadcast_to(theta_reg[f], th.shape)
            elif policy == "et_only":
                th = np.zeros_like(th)
            Ds = kp.forecast_dry(th, hs, t0, tp[f][None, :], inp, max_days=MAX_DAYS, dt=0.1)
            Wt = np.ones_like(Ds)
            if ask:
                # SmartSoil asks "how much water is in the field now?" where the answer could change the
                # advice (Experiment 13): the chance of irrigating within the lead time is between 20 and 80 %
                p_go = (Ds <= lead).mean(0)
                q = np.flatnonzero((p_go > 0.2) & (p_go < 0.8))
                if len(q):
                    fq = f[q]
                    cat = kp.depth_category(h_end[q])
                    flip = draws["ask_wet"][fq, d] > ASK_P_CORRECT
                    cat = np.where(flip, np.where(cat > 0, 0, 1), cat)
                    step = (draws["ask_cat"][fq, d] > ASK_CAT_CORRECT) & (cat > 0)
                    cat = np.where(step, np.clip(cat + np.where(draws["ask_cat"][fq, d] > (1 + ASK_CAT_CORRECT) / 2, 1, -1), 1, 3), cat)
                    Wt[:, q] = kp.answer_likelihood(hs[:, q], cat, "depth", p_correct=ASK_P_CORRECT, cat_correct=ASK_CAT_CORRECT)
                    asked[fq, d] = True
            D = kp.weighted_quantiles(Ds, Wt, [risk])[0]
            dry_now = (Wt * (hs > 0)).sum(0) / Wt.sum(0) <= 1 - risk
        D = np.where(dry_now, 0.0, D)
        go = D <= lead
        if not go.any():
            continue
        f, D = f[go], D[go]
        day_irr = np.maximum(np.ceil(t0 + D + wait - 1e-9) - s0, d + 1)
        j = np.minimum(n_irr[f], MAX_IRR - 1)
        next_irr[f] = s0 + day_irr + draws["advice_hour"][f, j]
        advice_day[f, d] = day_irr

    H = np.full((n, T), np.nan)
    H[:, :-1] = h6[:, ps.PER_DAY::ps.PER_DAY]
    H[:, -1] = h6[:, -1]
    return {"policy": policy, "H": np.where(season, H, np.nan), "h6": h6, "irr_mm": irr_mm, "n_irr": n_irr,
            "advised": advised, "advice_day": advice_day, "asked": asked, "season": season, "X": X}


def outcomes(arm, area_m2=None):
    """Per field: irrigation water (mm, after the puddling flood), irrigations, days ponded, days below the
    surface and days below the AWD safe threshold (end-of-day level)."""
    H, season = arm["H"], arm["season"]
    area = np.ones(len(H)) * 1000.0 if area_m2 is None else np.asarray(area_m2, float)
    lvl = np.where(season, H, np.nan)
    with np.errstate(invalid="ignore"):
        return {"water_mm": arm["irr_mm"].sum(1), "water_m3": arm["irr_mm"].sum(1) * area / 1000,
                "irrigations": arm["n_irr"].astype(float), "ponded_days": (lvl > 0).sum(1).astype(float),
                "dry_days": (lvl <= 0).sum(1).astype(float), "stress_days": (lvl < STRESS_MM).sum(1).astype(float),
                "season_days": season.sum(1).astype(float)}


def run(sim, cfg, inp, policies=("practice", "model", "perfect"), seed=0, wait=1, follow=1.0, model=None, n_samples=24,
        risk=0.5, lead=1.0, no_log=None, ask=False):
    draws = _shared_draws(sim, cfg, inp, seed)
    return {p: run_arm(sim, cfg, inp, p, seed, wait, follow, model, n_samples, draws, risk, lead, no_log, ask) for p in policies}
