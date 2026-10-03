"""Rice methane (src/awdrt/methane.py): drainage periods, regimes, IPCC Tier 1 values, satellite-only wet days."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import methane as mh  # noqa: E402

W, D = True, False


def periods(row, min_days=2):
    wet = np.array([row])
    return int(mh.drainage_periods(wet, np.ones_like(wet), min_days)[0])


def test_drainage_periods():
    assert periods([W] * 10) == 0
    assert periods([W, W, D, W, W]) == 0                      # one dry day is not a drainage period (2-day rule)
    assert periods([W, W, D, W, W], 1) == 1
    assert periods([W, D, D, W, W, D, D, D, W]) == 2
    assert periods([W, W, W, D, D, D]) == 0                   # end-of-season drainage does not count
    assert periods([D, D, D, W, W, W]) == 0                   # before the first flooding does not count
    assert periods([D] * 6) == 0
    assert list(mh.regime([0, 1, 2, 5])) == ["continuous", "single", "multiple", "multiple"]


def test_tier1_values():
    kg = mh.emissions(["continuous", "single", "multiple", "aggregated"], [100] * 4)
    assert kg == pytest.approx([85.0, 60.35, 46.75, 51.0])     # 0.85 kg/ha/d x SF_w x 100 days (IPCC 2019, South Asia)
    c, lo, hi = mh.uncertainty(["continuous", "multiple"], [100, 100], [1.0, 2.0])
    assert c == pytest.approx(85 + 2 * 46.75) and lo < c < hi
    assert 0.5 < lo / c < 0.8 and 1.3 < hi / c < 1.9            # the EF and SF ranges, shared across fields


def test_satellite_wet_takes_the_last_call():
    ch = {"s2_wet": 0, "s2_dry": 1, "s1_wet": 2, "s1_dry": 3}
    X = np.zeros((1, 6, 4))
    X[0, 1, 0] = 1           # S2 wet on day 1
    X[0, 3, 3] = 1           # S1 dry on day 3
    nisar = np.full((1, 6), -1, np.int8)
    nisar[0, 3] = 1          # NISAR wet at 06:00 on day 3, overridden by S1 dry at 18:00
    assert mh.satellite_wet(X, ch, nisar)[0].tolist() == [False, True, True, False, False, False]
