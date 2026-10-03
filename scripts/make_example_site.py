"""Write app/sites/example_irregular.geojson: a SYNTHETIC smallholder layout to exercise the site format.

It imitates what a plot survey around tubewells in north Bangladesh might return (long, narrow, slightly skewed
plots on both sides of earthen channels, three tubewells, about 20 water-level loggers), but it is not a survey:
the file says surveyed: false, and the app labels it as an example. Replace it with the campaign's surveyed
polygons and logger positions via scripts/site_from_survey.py.
"""

import json
import math
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "app" / "sites" / "example_irregular.geojson"
LON0, LAT0 = 89.985, 25.205      # placed inside the study window only so distances and weather are plausible
KX = math.radians(1) * 6_371_000 * math.cos(math.radians(LAT0))
KZ = math.radians(1) * 6_371_000


def ll(x, z):  # local metres (x east, z south) -> [lon, lat]
    return [round(LON0 + x / KX, 7), round(LAT0 - z / KZ, 7)]


def add_plot_attributes(feats):
    """SYNTHETIC plot attributes for the logger placement planner, from a separate random stream so the
    geometry stays as before: soil (a smooth pattern across the site, as soil maps are), whether the farmer
    has a smartphone, and whether the farmer agreed to a logger (asked in December)."""
    rng = np.random.default_rng(2028)
    ph = rng.uniform(0, 2 * math.pi, 3)
    for f in feats:
        p = f["properties"]
        if p["kind"] != "field":
            continue
        ring = f["geometry"]["coordinates"][0]
        x = (sum(c[0] for c in ring) / len(ring) - LON0) * KX
        z = -(sum(c[1] for c in ring) / len(ring) - LAT0) * KZ
        u = math.sin(x / 110 + ph[0]) + math.sin(z / 150 + ph[1]) + 0.6 * math.sin((x + z) / 70 + ph[2]) + rng.normal(0, 0.35)
        p["soil"] = "clay loam" if u < -0.6 else "silt loam" if u < 0.5 else "sandy loam"
        p["smartphone"] = "yes" if rng.random() < 0.75 else "no"
        r = rng.random()
        p["consent"] = "no" if r < 0.08 else "not asked" if r < 0.3 else "yes"


def main():
    rng = np.random.default_rng(2027)
    feats, field_id, logger_id = [], 0, 0
    for t in range(3):
        z_ch = t * 230.0                      # channel line of this command area (runs east, slight bend)
        bend = rng.uniform(-8, 8)
        ch = lambda x: z_ch + bend * math.sin(x / 140)  # noqa: E731
        x_end = rng.uniform(330, 400)
        feats.append({"type": "Feature", "properties": {"kind": "tubewell", "id": str(t + 1), "name": f"Tubewell {t + 1}"},
                      "geometry": {"type": "Point", "coordinates": ll(-14, ch(0))}})
        feats.append({"type": "Feature", "properties": {"kind": "channel", "tubewell": str(t + 1), "width_m": 2.5},
                      "geometry": {"type": "LineString", "coordinates": [ll(x, ch(x)) for x in np.linspace(-9, x_end, 12)]}})
        for side in (-1, 1):                  # plots north and south of the channel
            x = 0.0
            while x < x_end - 12:
                w = float(rng.uniform(14, 30))
                depth = float(rng.uniform(45, 95))
                skew = float(rng.uniform(-6, 6))
                x0, x1 = x, min(x + w, x_end)
                z_near0, z_near1 = ch(x0) + side * 2.5, ch(x1) + side * 2.5
                split = depth > 75 and rng.random() < 0.5   # long plots often split into two owners
                parts = [(0, depth / 2), (depth / 2, depth)] if split else [(0, depth)]
                for d0, d1 in parts:
                    j = lambda: float(rng.uniform(-1.5, 1.5))  # noqa: E731
                    poly = [[x0 + skew * d0 / depth + j(), z_near0 + side * d0], [x1 + skew * d0 / depth + j(), z_near1 + side * d0],
                            [x1 + skew * d1 / depth + j(), z_near1 + side * d1], [x0 + skew * d1 / depth + j(), z_near0 + side * d1]]
                    if side < 0:
                        poly = poly[::-1]
                    field_id += 1
                    fid = f"T{t + 1}-{field_id:03d}"
                    feats.append({"type": "Feature", "properties": {"kind": "field", "id": fid, "tubewell": str(t + 1)},
                                  "geometry": {"type": "Polygon", "coordinates": [[ll(*p) for p in poly + poly[:1]]]}})
                    if rng.random() < 0.2:            # about one plot in five carries a logger, near a corner
                        logger_id += 1
                        cx = (poly[0][0] + poly[1][0]) / 2 * 0.75 + (poly[2][0] + poly[3][0]) / 2 * 0.25
                        cz = (poly[0][1] + poly[1][1]) / 2 * 0.75 + (poly[2][1] + poly[3][1]) / 2 * 0.25
                        feats.append({"type": "Feature", "properties": {"kind": "logger", "id": f"L{logger_id:02d}", "field": fid},
                                      "geometry": {"type": "Point", "coordinates": ll(cx + 3, cz)}})
                x = x1 + 0.0
        if t < 2:
            feats.append({"type": "Feature", "properties": {"kind": "road", "width_m": 5},
                          "geometry": {"type": "LineString", "coordinates": [ll(-40, z_ch + 115), ll(420, z_ch + 112)]}})
    feats.append({"type": "Feature", "properties": {"kind": "road", "width_m": 6},
                  "geometry": {"type": "LineString", "coordinates": [ll(-32, -120), ll(-30, 580)]}})
    add_plot_attributes(feats)
    gj = {"type": "FeatureCollection",
          "properties": {"name": "example_irregular", "label": "Example: irregular smallholder plots (synthetic, not surveyed)",
                         "surveyed": False, "source": "synthetic, generated by scripts/make_example_site.py (plot attributes too); replace with the campaign survey"},
          "features": feats}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, indent=1))
    kinds = [f["properties"]["kind"] for f in feats]
    print(f"wrote {OUT}: {kinds.count('field')} fields, {kinds.count('logger')} loggers, {kinds.count('tubewell')} tubewells")


if __name__ == "__main__":
    main()
