"""PROSAIL simulation of transplanted rice over switching backgrounds -> Sentinel-2 bands."""

import numpy as np
import pandas as pd
import prosail

from . import backgrounds
from .s2 import FEATURE_BANDS, convolve


def sample_canopy(rng, lai_max=6.0):
    """Rice canopy parameters. LAI is drawn uniform so every stage is equally
    represented; the other traits follow published rice ranges."""
    return dict(
        n=rng.uniform(1.2, 2.0),
        cab=rng.uniform(20, 60),
        car=rng.uniform(4, 15),
        cbrown=rng.uniform(0.0, 0.2),
        cw=rng.uniform(0.008, 0.025),
        cm=rng.uniform(0.003, 0.008),
        lai=rng.uniform(0.0, lai_max),
        lidfa=rng.uniform(50, 75),  # rice is erectophile (ellipsoidal ALA)
        hspot=rng.uniform(0.05, 0.2),
        # Sentinel-2 over Bangladesh, Dec-May, ~10:30 local time.
        tts=rng.uniform(25, 55),
        tto=rng.uniform(0, 10),
        psi=rng.uniform(0, 180),
    )


def simulate(n, rng, classes=backgrounds.CLASSES, lai_max=6.0, noise=True):
    """Simulate n samples, backgrounds drawn evenly from `classes`.

    Returns a DataFrame with the canopy/background parameters, the class label
    and noisy S2 band reflectances (columns named after FEATURE_BANDS).
    """
    rows, spectra = [], np.empty((n, 2101))
    for i in range(n):
        cls = classes[i % len(classes)]
        bg, bgp = backgrounds.sample_background(cls, rng)
        c = sample_canopy(rng, lai_max)
        spectra[i] = prosail.run_prosail(
            c["n"], c["cab"], c["car"], c["cbrown"], c["cw"], c["cm"], max(c["lai"], 1e-3),
            c["lidfa"], c["hspot"], c["tts"], c["tto"], c["psi"],
            typelidf=2, rsoil0=bg, factor="SDR",
        )
        rows.append({**c, **{f"bg_{k}": v for k, v in bgp.items()}, "background": cls})
    R = convolve(spectra)
    if noise:
        # L2A uncertainty: ~3 % relative plus a 0.005 absolute floor (atmospheric
        # correction residuals dominate over dark water in the SWIR).
        R = R * (1 + rng.normal(0, 0.03, R.shape)) + rng.normal(0, 0.005, R.shape)
    df = pd.DataFrame(rows)
    df[FEATURE_BANDS] = np.clip(R, 0, None)
    df["cwc"] = df["lai"] * df["cw"]  # canopy water content, g/cm2 (LAI x Cw)
    return df
