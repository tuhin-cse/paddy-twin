"""Fetch ERA5 at the Jamalpur/Sherpur site and compute daily FAO-56 ET0 and rain."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awdrt import era5  # noqa: E402

LAT, LON = 25.205, 89.985
h = era5.hourly(LAT, LON, "2023-12-01", "2024-04-30", cache_dir=ROOT / "data" / "cache" / "era5")
d = era5.fao56_daily(h, LAT)
d.loc["2023-12-01":"2024-04-30"].round(3).to_csv(ROOT / "results" / "era5_daily_jamalpur.csv")
print(d[["et0", "rain", "tmean", "rs"]].resample("MS").agg(["mean", "sum"]).round(2))
