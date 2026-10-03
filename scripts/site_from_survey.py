"""Build a Paddy Twin site file (app/sites/<name>.geojson) from a field survey.

Inputs (WGS84 lon/lat):
  --fields     plot boundaries: GeoJSON (Polygon/MultiPolygon features) or KML (Placemarks with Polygons),
               e.g. a GPS walk of each bund, or farmer-drawn boundaries exported from SmartSoil.
               An "id" (GeoJSON property, or the KML Placemark name) is kept as the field id.
  --loggers    CSV with columns id, lat, lon [, field]: the water-level loggers (AWD pipes) as installed.
               A logger without a field is assigned to the plot that contains it.
  --tubewells  CSV with columns id, lat, lon [, name]. Each field is served by its nearest tubewell unless
               the fields file gives a "tubewell" property.
  --attributes optional CSV with a field_id column and one column per plot attribute (soil, smartphone,
               consent, ...): the December plot survey sheet. The logger placement planner (app/planner.py) uses
               every attribute as a stratum; consent = no keeps a plot out of any plan. Properties of GeoJSON
               plot features are kept too.
  --channels   optional GeoJSON with LineString features (property "tubewell"); roads may be added the same
               way with property kind = "road".
Example:
  python scripts/site_from_survey.py --name jamalpur_2027 --label "Jamalpur campaign plots (surveyed Dec 2026)" \\
      --fields survey/plots.kml --loggers survey/loggers.csv --tubewells survey/tubewells.csv --surveyed
Checks: every polygon is closed and non-degenerate; every logger falls inside a plot; ids are unique.
"""

import argparse
import csv
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import sites  # noqa: E402


def read_fields(path):
    path = pathlib.Path(path)
    out = []
    if path.suffix.lower() == ".kml":
        root = ET.parse(path).getroot()
        ns = {"k": re.match(r"\{.*\}", root.tag).group(0)[1:-1]} if root.tag.startswith("{") else {"k": ""}
        q = (lambda t: f"k:{t}") if ns["k"] else (lambda t: t)
        for i, pm in enumerate(root.iter(f"{{{ns['k']}}}Placemark" if ns["k"] else "Placemark")):
            name = pm.find(q("name"), ns)
            for poly in pm.iter(f"{{{ns['k']}}}Polygon" if ns["k"] else "Polygon"):
                coords = poly.find(f".//{q('outerBoundaryIs')}//{q('coordinates')}", ns)
                ring = [[float(v) for v in tok.split(",")[:2]] for tok in coords.text.split()]
                out.append({"id": (name.text.strip() if name is not None and name.text else f"F{i + 1}"), "ring": ring, "props": {}})
    else:
        for i, f in enumerate(json.loads(path.read_text())["features"]):
            g, props = f["geometry"], f.get("properties") or {}
            rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]] if g["type"] == "MultiPolygon" else []
            for k, ring in enumerate(rings):
                fid = str(props.get("id", props.get("name", f"F{i + 1}")))
                out.append({"id": fid if len(rings) == 1 else f"{fid}-{k + 1}", "ring": [c[:2] for c in ring], "props": props})
    return out


def read_points(path):
    with open(path, newline="") as fh:
        return [{k.strip().lower(): v.strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="file name in app/sites (letters, digits, _ or -)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--loggers")
    ap.add_argument("--tubewells")
    ap.add_argument("--channels")
    ap.add_argument("--attributes", help="CSV: field_id plus plot attributes (soil, smartphone, consent, ...)")
    ap.add_argument("--surveyed", action="store_true", help="mark as surveyed field data (not an example)")
    ap.add_argument("--source", default="field survey")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.name):
        sys.exit("--name: use letters, digits, _ or - only")
    feats = []
    fields = read_fields(a.fields)
    attrs = {}
    for row in read_points(a.attributes) if a.attributes else []:
        fid = row.pop("field_id", None)
        if fid:
            attrs[fid] = {k: v for k, v in row.items() if k and k not in ("kind", "id", "tubewell")}
    ids = [f["id"] for f in fields]
    if len(set(ids)) != len(ids):
        sys.exit("field ids are not unique")
    for f in fields:
        ring = f["ring"] if f["ring"][0] == f["ring"][-1] else f["ring"] + f["ring"][:1]
        props = {"kind": "field", "id": f["id"]}
        props.update({k: v for k, v in f["props"].items() if k not in ("kind", "id", "name", "tubewell") and isinstance(v, (str, int, float, bool))})
        props.update(attrs.pop(f["id"], {}))
        if "tubewell" in f["props"]:
            props["tubewell"] = str(f["props"]["tubewell"])
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    if attrs:
        print(f"warning: --attributes rows for unknown plots ignored: {sorted(attrs)[:5]}", file=sys.stderr)
    for t in read_points(a.tubewells) if a.tubewells else []:
        feats.append({"type": "Feature", "properties": {"kind": "tubewell", "id": t["id"], "name": t.get("name") or f"Tubewell {t['id']}"},
                      "geometry": {"type": "Point", "coordinates": [float(t["lon"]), float(t["lat"])]}})
    for lg in read_points(a.loggers) if a.loggers else []:
        props = {"kind": "logger", "id": lg["id"]}
        if lg.get("field"):
            props["field"] = lg["field"]
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [float(lg["lon"]), float(lg["lat"])]}})
    if a.channels:
        for f in json.loads(pathlib.Path(a.channels).read_text())["features"]:
            if f["geometry"]["type"] == "LineString":
                p = f.get("properties") or {}
                kind = "road" if p.get("kind") == "road" else "channel"
                feats.append({"type": "Feature", "properties": {**p, "kind": kind}, "geometry": f["geometry"]})
    out = sites.SITES_DIR / f"{a.name}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection",
                               "properties": {"name": a.name, "label": a.label, "surveyed": a.surveyed, "source": a.source},
                               "features": feats}, indent=1))
    lay = sites.load_site(a.name)   # validates polygons and logger placement
    print(f"wrote {out}: {len(lay['fields'])} fields ({sum(f['area_m2'] for f in lay['fields']) / 1e4:.1f} ha), "
          f"{len(lay['loggers'])} loggers, {len(lay['tubewells'])} tubewells, {len(lay['channels'])} channels")


if __name__ == "__main__":
    main()
