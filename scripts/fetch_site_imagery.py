"""Fetch real Sentinel-2 true-colour images of a site for Paddy Twin's 3D village and 2D map.

For a site file (its plot centroid) or the study area (default, used under the generated grid), this finds clear
Sentinel-2 L2A scenes of the 2024 Boro season on Element 84's Earth Search (AWS Open Data), reads the 10 m
true-colour band (TCI) around the site, reprojects it to the site's local metres (azimuthal equidistant around the
centre, x east, north up; the frame app/sites.py uses) and writes one JPEG per date with a meta.json:

  app/sites/imagery/<site>/YYYY-MM-DD.jpg
  app/sites/imagery/<site>/meta.json   {center_lonlat, half_size_m, px, dates, source, attribution}

The images are real; a synthetic site's plots are not, so the app says so. Contains modified Copernicus Sentinel
data 2024.

  python scripts/fetch_site_imagery.py                       # the study area (generated grid)
  python scripts/fetch_site_imagery.py --site example_irregular
"""

import argparse
import datetime as dt
import json
import pathlib
import sys
import urllib.request

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import sites  # noqa: E402

OUT = ROOT / "app" / "sites" / "imagery"
SEARCH = "https://earth-search.aws.element84.com/v1/search"
STUDY_CENTER = (89.985, 25.205)          # centre of the study window (scripts/02_real_s2_boro_jamalpur.py)
SEASON = ("2024-01-05", "2024-04-25")    # the simulated Boro season (15 Jan - 15 Apr 2024) with a margin


def search(lon, lat, start, end, max_cloud):
    body = {"collections": ["sentinel-2-l2a"], "bbox": [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 100}
    req = urllib.request.Request(SEARCH, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    feats = json.load(urllib.request.urlopen(req, timeout=60))["features"]
    by_date = {}
    for f in feats:
        by_date.setdefault(f["properties"]["datetime"][:10], []).append(f)
    return by_date


def read_window(href, lon, lat, half, px):
    """The TCI around (lon, lat), reprojected to a north-up local grid of 2*half metres and px pixels."""
    dst_crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
    dst_transform = rasterio.transform.from_bounds(-half, -half, half, half, px, px)
    out = np.zeros((3, px, px), np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(href) as src:
            b = transform_bounds(dst_crs, src.crs, -half * 1.1, -half * 1.1, half * 1.1, half * 1.1)
            win = rasterio.windows.from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            data = src.read(window=win, boundless=True, fill_value=0)
            reproject(data, out, src_transform=src.window_transform(win), src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="site file name in app/sites (default: the study area)")
    ap.add_argument("--half", type=float, default=3000.0, help="half the image width, metres")
    ap.add_argument("--px", type=int, default=600, help="image size, pixels (600 over 6 km = Sentinel-2's 10 m)")
    ap.add_argument("--every", type=int, default=12, help="target spacing between dates, days")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    a = ap.parse_args()
    if a.site:
        lon, lat = sites.load_site(a.site)["origin_lonlat"]
        key = a.site
    else:
        lon, lat = STUDY_CENTER
        key = "_study_area"
    by_date = search(lon, lat, *SEASON, a.max_cloud)
    d0, d1 = (dt.date.fromisoformat(s) for s in SEASON)
    targets = [d0 + dt.timedelta(days=k) for k in range(0, (d1 - d0).days + 1, a.every)]
    avail = sorted(dt.date.fromisoformat(s) for s in by_date)
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for t in targets:
        for cand in sorted(avail, key=lambda x: abs((x - t).days))[:4]:
            if cand.isoformat() in done or abs((cand - t).days) > a.every:
                continue
            best = None
            for f in by_date[cand.isoformat()]:
                img = read_window(f["assets"]["visual"]["href"], lon, lat, a.half, a.px)
                empty = float((img.max(0) == 0).mean())
                cloud = float(((img > 200).all(0)).mean())
                if best is None or empty < best[0]:
                    best = (empty, cloud, img, f["id"])
            if best and best[0] < 0.01 and best[1] < 0.03:
                Image.fromarray(np.moveaxis(best[2], 0, -1)).save(out_dir / f"{cand.isoformat()}.jpg", quality=82)
                done.append(cand.isoformat())
                print(f"{cand}: {best[3]} (cloud-like {best[1]:.1%})")
                break
    done.sort()
    meta = {"center_lonlat": [lon, lat], "half_size_m": a.half, "px": a.px, "dates": done,
            "source": "Sentinel-2 L2A true colour (TCI, 10 m), Element 84 Earth Search on AWS Open Data",
            "attribution": "Contains modified Copernicus Sentinel data 2024", "site": key}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(done)} images to {out_dir}")


if __name__ == "__main__":
    main()
