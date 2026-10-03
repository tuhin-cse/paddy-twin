"""awdrt: background-aware canopy RT retrieval for rice water-management (AWD) monitoring.

Early-season rice sits on a background that switches between standing water,
saturated mud and drying soil. Standard hybrid retrievals (PROSAIL + ML, the
SNAP biophysical processor) assume a soil background. This package simulates
Sentinel-2 reflectance for rice over all three backgrounds and trains a joint
retriever for LAI *and* background state.
"""
