"""Sentinel-2 MSI band definitions and spectral convolution.

The spectral response functions are approximated as Gaussians with the
published S2A centre wavelengths and bandwidths (ESA S2 MSI User Handbook).
That is adequate for a feasibility study; a publication-grade run should use
the official SRF tables (S2-SRF_COPE-GSEG-EOPG-TN-15-0007).
"""

import numpy as np

WL = np.arange(400, 2501)  # PROSAIL grid, 1 nm

# band: (centre nm, FWHM nm, native resolution m)
BANDS = {
    "B02": (492.4, 66, 10),
    "B03": (559.8, 36, 10),
    "B04": (664.6, 31, 10),
    "B05": (704.1, 15, 20),
    "B06": (740.5, 15, 20),
    "B07": (782.8, 20, 20),
    "B08": (832.8, 106, 10),
    "B8A": (864.7, 21, 20),
    "B11": (1613.7, 91, 20),
    "B12": (2202.4, 175, 20),
}

# The bands the retriever uses. B08 is left out because B8A carries the same
# NIR information at the 20 m grid everything else is on.
FEATURE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]


def _srf_matrix(bands):
    m = np.zeros((len(bands), WL.size))
    for i, b in enumerate(bands):
        c, fwhm, _ = BANDS[b]
        sigma = fwhm / 2.3548
        w = np.exp(-0.5 * ((WL - c) / sigma) ** 2)
        m[i] = w / w.sum()
    return m


SRF = _srf_matrix(FEATURE_BANDS)


def convolve(spectra):
    """(n, 2101) 1-nm spectra -> (n, len(FEATURE_BANDS)) band reflectance."""
    return np.atleast_2d(spectra) @ SRF.T


def band_index(name):
    return FEATURE_BANDS.index(name)


def indices(R):
    """Standard indices from an (n, bands) array in FEATURE_BANDS order."""
    b = {k: R[..., band_index(k)] for k in FEATURE_BANDS}
    eps = 1e-6
    ndvi = (b["B8A"] - b["B04"]) / (b["B8A"] + b["B04"] + eps)
    evi = 2.5 * (b["B8A"] - b["B04"]) / (b["B8A"] + 6 * b["B04"] - 7.5 * b["B02"] + 1)
    lswi = (b["B8A"] - b["B11"]) / (b["B8A"] + b["B11"] + eps)
    mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + eps)
    return {"ndvi": ndvi, "evi": evi, "lswi": lswi, "mndwi": mndwi}
