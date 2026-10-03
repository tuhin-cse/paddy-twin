"""Rice methane from each field's water history: IPCC Tier 1 (2019 Refinement), field by field.

    CH4 (kg/ha/season) = EF_c x SF_w x SF_p x SF_o x days of cultivation

  EF_c  baseline: continuously flooded, not flooded < 180 days before the season, no organic amendments.
        2019 Refinement Table 5.11, South Asia: 0.85 kg CH4/ha/day (95 % range 0.58-1.26).
  SF_w  water regime during cultivation, Table 5.12 (disaggregated, irrigated):
          continuously flooded 1.00 (0.73-1.27)  standing water all season, dry only for harvest
          single drainage      0.71 (0.53-0.94)  one drainage period besides the end-of-season drainage
          multiple drainage    0.55 (0.41-0.72)  more than one, including alternate wetting and drying
  SF_p  water before the season, Table 5.13: Boro follows the Aman harvest with a dry fallow of less than
        180 days, so 1.00 (0.88-1.12).
  SF_o  organic amendments, Table 5.14: none assumed (1.00).
Values checked against the IPCC 2019 Refinement presentation (WGIA, 2019); cite the Refinement itself.

  (aggregated) the Table 5.12 "irrigated" factor for an inventory that does not know the regime: 0.60 (0.44-0.78)

Neither the Refinement nor carbon-credit methodologies built on it (e.g. Gold Standard's AWD methodology) fix how
long a drainage period must last. Here it is at least `MIN_DRY_DAYS` (2) consecutive in-season days without
standing water at the end of the day; a dry spell that runs to the last day of the record is the end-of-season
drainage and does not count. A field's regime then follows from its number of drainage periods (0, 1, 2 or
more). The choice matters: in simulator v2 the level falls from the surface to -15 cm within a day, so AWD
cycles are short, and with 3 days many AWD fields would count as continuously flooded.

The regime can come from different views of the same field: the true (or logged) water level, KGML-Paddy's daily
belief (ponded when P(ponded) > 0.5), the satellite calls alone (each day takes the last call), or one regional
default for every field (what an inventory without field data assumes). `uncertainty` propagates the IPCC 95 %
ranges by Monte Carlo, with each factor shared by all fields (an inventory's error is systematic, not per field).

GWP100 of biogenic methane: 27.0 (IPCC AR6, WG1 Table 7.15).
"""

import numpy as np

EF_C = {"South Asia": (0.85, 0.58, 1.26), "Southeast Asia": (1.22, 0.83, 1.81), "East Asia": (1.32, 0.89, 1.96),
        "Global": (1.19, 0.80, 1.76)}
REGIMES = ("continuous", "single", "multiple", "aggregated")
REGIME_NAMES = {"continuous": "continuously flooded", "single": "single drainage", "multiple": "multiple drainage (incl. AWD)",
                "aggregated": "irrigated, regime unknown (IPCC aggregated)"}
SF_W = {"continuous": (1.00, 0.73, 1.27), "single": (0.71, 0.53, 0.94), "multiple": (0.55, 0.41, 0.72),
        "aggregated": (0.60, 0.44, 0.78)}
SF_P = (1.00, 0.88, 1.12)
GWP100 = 27.0
MIN_DRY_DAYS = 2


def drainage_periods(wet, season, min_days=MIN_DRY_DAYS):
    """Number of drainage periods per field: runs of >= min_days in-season days without standing water,
    after the field was first flooded, not counting a run that reaches the end of the record. wet, season: (n, T)."""
    wet, season = np.asarray(wet, bool), np.asarray(season, bool)
    n, T = wet.shape
    out = np.zeros(n, int)
    for f in range(n):
        idx = np.flatnonzero(season[f])
        if not len(idx):
            continue
        w = wet[f, idx]
        if not w.any():
            continue
        w = w[np.argmax(w):]                       # from the first flooded day
        dry = np.r_[False, ~w, False]
        edges = np.flatnonzero(dry[1:] != dry[:-1])
        for a, b in zip(edges[::2], edges[1::2]):  # run [a, b)
            if b - a >= min_days and b < len(w):
                out[f] += 1
    return out


def regime(periods):
    p = np.asarray(periods)
    return np.where(p == 0, "continuous", np.where(p == 1, "single", "multiple"))


def emissions(regimes, days, region="South Asia"):
    """kg CH4/ha/season per field (central Tier 1 values)."""
    sf = np.array([SF_W[r][0] for r in regimes])
    return EF_C[region][0] * sf * SF_P[0] * np.asarray(days, float)


def _lognormal(rng, central, lo, hi, size):
    """Draws around the central value, log-normal with the 95 % range's width."""
    s = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    return central * np.exp(s * rng.standard_normal(size))


def uncertainty(regimes, days, area_ha, region="South Asia", n=4000, seed=0):
    """Village total (kg CH4) central value and 95 % range, from the IPCC factor ranges (log-normal, shared)."""
    rng = np.random.default_rng(seed)
    ef = _lognormal(rng, *EF_C[region], n)
    sp = _lognormal(rng, *SF_P, n)
    sw = {r: _lognormal(rng, *SF_W[r], n) for r in REGIMES}
    days, area = np.asarray(days, float), np.asarray(area_ha, float)
    tot = np.zeros(n)
    for r in REGIMES:
        m = np.asarray(regimes) == r
        tot += sw[r] * float((days[m] * area[m]).sum())
    tot *= ef * sp
    central = float(emissions(regimes, days, region) @ area)
    lo, hi = np.quantile(tot, [0.025, 0.975])
    return central, float(lo), float(hi)


def satellite_wet(X, ch, extra=None):
    """Daily wet/dry from the satellite calls alone: each day takes the most recent call (dry before the first).
    X: model inputs (Sentinel-2 and Sentinel-1 channels); extra: optional (n, T) int8 calls of another sensor
    (-1 none, 0 dry, 1 wet), e.g. NISAR at 06:00, which comes before both."""
    n, T = X.shape[:2]
    state = np.full((n, T), np.nan)
    if extra is not None:
        state = np.where(extra >= 0, extra.astype(float), state)
    for s in ("s2", "s1"):                          # 10:30, then 18:00: the later call of the day wins
        w, d = X[:, :, ch[f"{s}_wet"]] > 0, X[:, :, ch[f"{s}_dry"]] > 0
        state = np.where(w, 1.0, np.where(d, 0.0, state))
    out = np.zeros((n, T), bool)
    last = np.full(n, np.nan)
    for t in range(T):
        last = np.where(np.isfinite(state[:, t]), state[:, t], last)
        out[:, t] = last > 0.5
    return out


def summarise(wet, season, area_ha, region="South Asia"):
    """Per field: drainage periods, regime, season days, kg CH4/ha; and the village total with its range."""
    season = np.asarray(season, bool)
    per = drainage_periods(wet, season)
    reg = regime(per)
    days = season.sum(1)
    kg = emissions(reg, days, region)
    c, lo, hi = uncertainty(reg, days, area_ha, region)
    return {"periods": per, "regime": reg, "days": days, "kg_ha": kg, "total_kg": c, "total_lo": lo, "total_hi": hi,
            "ponded_days": (np.asarray(wet, bool) & season).sum(1)}
