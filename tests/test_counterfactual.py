"""Advice against practice (src/awdrt/counterfactual.py) and its Paddy Twin endpoint."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import counterfactual as cf  # noqa: E402
from awdrt import paddy_sim as ps  # noqa: E402


@pytest.fixture(scope="module")
def village():
    inp = ps.load_inputs()
    cfg = ps.WORLDS["prior"]
    return cfg, inp, ps.simulate(cfg, 60, 11, inp)


def test_practice_reproduces_the_simulator(village):
    cfg, inp, sim = village
    A = cf.run_arm(sim, cfg, inp, "practice")
    assert A["n_irr"].mean() == pytest.approx(sim["n_irr"].mean(), rel=0.08)
    assert A["irr_mm"].sum(1).mean() == pytest.approx(sim["irr_true_mm"].mean(), rel=0.08)
    # the same physics: days ponded and below the surface match the simulator's season on average
    H = ps.end_of_day_depth(sim["h6"])
    o = cf.outcomes(A)
    season = A["season"]
    assert o["ponded_days"].mean() == pytest.approx(np.nansum(np.where(season, H, np.nan) > 0, 1).mean(), rel=0.08)

def test_arms_share_everything_but_the_policy(village):
    cfg, inp, sim = village
    draws = cf._shared_draws(sim, cfg, inp, 0)
    a = cf.run_arm(sim, cfg, inp, "practice", draws=draws)
    b = cf.run_arm(sim, cfg, inp, "practice", draws=draws)
    assert np.array_equal(a["irr_mm"], b["irr_mm"]) and np.array_equal(a["X"], b["X"])
    # nobody follows: the advice arm is the practice arm
    c = cf.run_arm(sim, cfg, inp, "perfect", draws=draws, follow=0.0)
    assert np.array_equal(a["irr_mm"], c["irr_mm"])


def test_perfect_advice_cuts_stress(village):
    cfg, inp, sim = village
    draws = cf._shared_draws(sim, cfg, inp, 0)
    p = cf.outcomes(cf.run_arm(sim, cfg, inp, "practice", draws=draws))
    q = cf.outcomes(cf.run_arm(sim, cfg, inp, "perfect", draws=draws, wait=0))
    assert q["stress_days"].mean() < 0.6 * p["stress_days"].mean()
    r = cf.outcomes(cf.run_arm(sim, cfg, inp, "perfect", draws=draws, wait=2))
    assert r["water_mm"].mean() < q["water_mm"].mean()      # waiting longer saves water
    assert r["stress_days"].mean() > q["stress_days"].mean()  # ... at the cost of dry stress
