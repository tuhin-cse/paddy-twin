"""Background reflectance under a rice canopy: ponded water, saturated mud, drying soil.

Ponded water follows the shallow-water semi-analytical model of Lee et al.
(1998, 1999): a turbid (Case II) water column of depth H over a wet-soil bottom,
with sky reflection at the surface. The water is treated as Lambertian because
4SAIL only accepts a Lambertian background. That is exactly the approximation
aquatic-vegetation radiative-transfer (AVRT/AVRM) models remove (specular glint, submerged stems, row gaps), so
the numbers here are a lower bound on what a proper aquatic-canopy model gives.

Soil uses the two PROSAIL end-members (dry, wet), mixed by `psoil`:
    rsoil0 = rsoil * (psoil * dry + (1 - psoil) * wet)
"""

import os

import numpy as np
import prosail

from .s2 import WL

_SOIL = np.loadtxt(os.path.join(os.path.dirname(prosail.__file__), "soil_reflectance.txt"))
SOIL_DRY, SOIL_WET = _SOIL[:, 0], _SOIL[:, 1]

# Pure-water absorption (1/m). Visible from Pope & Fry (1997); NIR/SWIR from
# Hale & Querry (1973) and Kou et al. (1993). Log-interpolated between knots, so
# fine structure is smoothed; the magnitude is what matters here.
_AW_KNOTS = np.array([
    (400, 0.00663), (450, 0.0092), (500, 0.0204), (550, 0.0565), (600, 0.2224),
    (650, 0.340), (700, 0.650), (725, 1.69), (750, 2.62), (775, 2.44), (800, 2.07),
    (825, 2.8), (850, 4.3), (875, 5.3), (900, 6.8), (925, 14), (950, 39), (975, 45),
    (1000, 36), (1050, 15), (1100, 17), (1150, 90), (1200, 104), (1250, 90),
    (1300, 115), (1350, 250), (1400, 1230), (1450, 3100), (1500, 1900), (1550, 950),
    (1600, 650), (1650, 560), (1700, 600), (1750, 850), (1800, 1100), (1850, 4000),
    (1900, 11000), (1950, 11000), (2000, 7000), (2050, 3300), (2100, 2300),
    (2150, 1800), (2200, 1600), (2250, 1800), (2300, 2300), (2350, 3000),
    (2400, 4000), (2450, 6000), (2500, 8000),
])
A_WATER = np.exp(np.interp(WL, _AW_KNOTS[:, 0], np.log(_AW_KNOTS[:, 1])))


def soil(psoil, rsoil):
    """Soil background; psoil=0 is wet (puddled mud), psoil=1 is dry."""
    return rsoil * (psoil * SOIL_DRY + (1 - psoil) * SOIL_WET)


def ponded_water(depth_m, tss_gm3, acdom440, bottom, sky_refl):
    """Above-surface reflectance factor of shallow turbid water over `bottom`.

    depth_m   water depth (paddy: 0.01-0.15 m)
    tss_gm3   total suspended solids (g/m3); paddy water is muddy, 10-200
    acdom440  CDOM absorption at 440 nm (1/m)
    bottom    bottom reflectance spectrum (2101,), normally wet soil
    sky_refl  Fresnel-reflected sky at 550 nm, in reflectance-factor units
    """
    a_nap = 0.035 * tss_gm3 * np.exp(-0.0123 * (WL - 443))
    a_cdom = acdom440 * np.exp(-0.015 * (WL - 440))
    a = A_WATER + a_nap + a_cdom
    bbw = 0.5 * 0.0038 * (400 / WL) ** 4.32
    bbp = 0.012 * tss_gm3 * (555 / WL) ** 0.5
    bb = bbw + bbp
    kappa = a + bb
    u = bb / kappa
    rrs_deep = (0.084 + 0.17 * u) * u
    # Lee et al. (1998) shallow-water form with nadir path-length factors.
    rrs = rrs_deep * (1 - np.exp(-2.1 * kappa * depth_m)) \
        + (bottom / np.pi) * np.exp(-2.2 * kappa * depth_m)
    Rrs = 0.52 * rrs / (1 - 1.7 * rrs)
    return np.pi * Rrs + sky_refl * (550 / WL) ** 1.5


# Background classes used throughout. AWD management turns on the first
# distinction: is there standing water (ponded) or has the field drained.
CLASSES = ["ponded", "saturated", "drying"]


def sample_background(cls, rng):
    """Draw one background spectrum of class `cls` plus its parameters."""
    if cls == "ponded":
        p = dict(depth=rng.uniform(0.01, 0.15), tss=np.exp(rng.uniform(np.log(5), np.log(250))),
                 acdom=rng.uniform(0.2, 3.0), rsoil=rng.uniform(0.5, 1.2),
                 sky=rng.uniform(0.003, 0.012))
        bottom = soil(0.0, p["rsoil"])
        return ponded_water(p["depth"], p["tss"], p["acdom"], bottom, p["sky"]), p
    if cls == "saturated":
        p = dict(psoil=rng.uniform(0.0, 0.2), rsoil=rng.uniform(0.5, 1.3))
    elif cls == "drying":
        p = dict(psoil=rng.uniform(0.2, 1.0), rsoil=rng.uniform(0.6, 1.6))
    else:
        raise ValueError(cls)
    return soil(p["psoil"], p["rsoil"]), p
