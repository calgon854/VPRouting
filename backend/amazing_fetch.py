import xarray as xr
import numpy as np
import pandas as pd

# --- Load the already-fetched dataset ---
ds = xr.open_dataset("total_0-33.nc")
da_all      = ds["TOT_PREC"]
hourly_rain = ds["hourly_rain"]

# --- Your 10,000 locations ---
# e.g. from a CSV with columns: name, lat, lon
locations = pd.read_csv("locations.csv")  # or build manually
lats = locations["lat"].values  # shape (N,)
lons = locations["lon"].values  # shape (N,)

# --- Vectorized nearest-neighbour lookup ---
grid_lat = da_all.lat.values  # shape (y, x)
grid_lon = da_all.lon.values  # shape (y, x)

# Broadcast: compute distance from every grid cell to every location at once
# grid_lat[:,:,None] is (y,x,1), lats[None,None,:] is (1,1,N) → dist is (y,x,N)
dist = np.sqrt(
    (grid_lat[:, :, None] - lats[None, None, :]) ** 2 +
    (grid_lon[:, :, None] - lons[None, None, :]) ** 2
)  # shape: (y, x, N)

# For each location find the (iy, ix) of the nearest cell
flat_idx = dist.reshape(-1, len(lats)).argmin(axis=0)   # shape (N,)
iy, ix = np.unravel_index(flat_idx, grid_lat.shape)     # each shape (N,)

# --- Extract values for all locations at once ---
# Result shapes: (N, lead_time) or (N, eps, lead_time) etc.
rain_all = hourly_rain.values[:, iy, ix]        # (lead_time, N)  — mean precip per hour
tot_all  = da_all.values[:, :, :, iy, ix]       # (eps, ref_time, lead_time, N) — adjust dim order to yours

# --- Build a tidy results DataFrame (example: hourly rain per location) ---
lead_times = np.arange(len(ds.lead_time))
result = pd.DataFrame(
    rain_all.T,                          # (N, lead_time)
    index=locations["name"],
    columns=[f"+{h:02d}h" for h in lead_times]
)
print(result.head())
result.to_csv("rain_all_locations.csv")