"""Sentinel-2 L2A access from two mirrors of the same ESA product.

- "pc":  Microsoft Planetary Computer STAC (collection sentinel-2-l2a), signed
         with the `planetary_computer` package. Also exposes sentinel-1-rtc.
- "aws": Element 84 Sentinel-2 COGs on AWS Open Data, read directly from the
         public bucket. Used when the Planetary Computer is unreachable.

Both return items as dicts {id, datetime, cloud, hrefs{band: url}, scale,
offset} so the analysis code does not care where the pixels came from.

Radiometric offset: from processing baseline 04.00 (25 Jan 2022) L2A digital
numbers carry +1000. Reflectance = DN * 1e-4 - 0.1 for those scenes on the
Planetary Computer, which serves ESA's digital numbers unchanged; the offset is
derived from s2:processing_baseline. The Element 84 COGs already have the
offset removed (see aws_items).
"""

import datetime as dt
import json
import re
import urllib.request

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]
AWS_BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def _baseline_offset(baseline):
    try:
        return -0.1 if float(baseline) >= 4.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── AWS (Element 84 COG bucket) ───────────────────────────────────────────────

def aws_items(mgrs_tile, start, end):
    """List L2A items for an MGRS tile such as '45RYH' between two dates."""
    utm, band, sq = mgrs_tile[:2], mgrs_tile[2], mgrs_tile[3:]
    out = []
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        prefix = f"sentinel-s2-l2a-cogs/{int(utm)}/{band}/{sq}/{y}/{m}/"
        xml = _get(f"{AWS_BUCKET}/?list-type=2&delimiter=/&prefix={prefix}")
        for p in re.findall(r"<Prefix>([^<]+_L2A/)</Prefix>", xml):
            item_id = p.rstrip("/").split("/")[-1]
            date = dt.datetime.strptime(item_id.split("_")[2], "%Y%m%d").date()
            if d0 <= date <= d1:
                out.append((date, f"{AWS_BUCKET}/{p}", item_id))
        m += 1
        if m == 13:
            y, m = y + 1, 1

    items = []
    for date, base, item_id in sorted(out):
        meta = json.loads(_get(f"{base}{item_id}.json"))
        props = meta["properties"]
        # Element 84 re-writes the COGs with the +1000 already removed
        # (earthsearch:boa_offset_applied). Its raster:bands still advertise
        # offset -0.1, and applying that on top drives dark pixels negative:
        # checked on this tile, where B02 DN goes as low as ~60.
        offset = 0.0 if props.get("earthsearch:boa_offset_applied") \
            else _baseline_offset(props.get("s2:processing_baseline"))
        items.append({
            "id": item_id,
            "date": date.isoformat(),
            "cloud": props.get("eo:cloud_cover"),
            "hrefs": {b: f"{base}{b}.tif" for b in BANDS},
            "scale": 1e-4,
            "offset": offset,
        })
    return items


# ── Planetary Computer ────────────────────────────────────────────────────────

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def pc_items(bbox, start, end, max_cloud=80, mgrs_tile=None):
    """Sentinel-2 L2A items from the Planetary Computer (signed hrefs)."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    query = {"eo:cloud_cover": {"lt": max_cloud}}
    if mgrs_tile:
        query["s2:mgrs_tile"] = {"eq": mgrs_tile}
    search = cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                        datetime=f"{start}/{end}", query=query)
    items = []
    for it in sorted(search.items(), key=lambda i: i.datetime):
        items.append({
            "id": it.id,
            "date": it.datetime.date().isoformat(),
            "cloud": it.properties.get("eo:cloud_cover"),
            "hrefs": {b: it.assets[b].href for b in BANDS},
            "scale": 1e-4,
            "offset": _baseline_offset(it.properties.get("s2:processing_baseline")),
        })
    return items


def pc_s1_rtc_items(bbox, start, end):
    """Sentinel-1 RTC (gamma0, linear power) items from the Planetary Computer."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}")
    items = []
    for it in sorted(search.items(), key=lambda i: i.datetime):
        items.append({
            "id": it.id,
            "date": it.datetime.date().isoformat(),
            "orbit": it.properties.get("sat:orbit_state"),
            "rel_orbit": it.properties.get("sat:relative_orbit"),
            "hrefs": {p: it.assets[p.lower()].href for p in ("VV", "VH") if p.lower() in it.assets},
        })
    return items
