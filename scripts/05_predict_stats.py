import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import re
import sys

# -------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------

prm_module_name = os.environ.get('PRM_MODULE')

if prm_module_name is None:
    raise ValueError("PRM_MODULE environment variable not set")

# Load the prm module from file using importlib.util
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prm_file = os.path.join(BASE_DIR, 'prm', f"{prm_module_name}.py")

if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Could not find prm file: {prm_file}")

spec = importlib.util.spec_from_file_location("prm", prm_file)
prm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prm)

cube_dir = prm.PREDICTIONS_DIR
products = prm.PRODUCTS
output_dir = Path(cube_dir) / "qc_plots"
output_dir.mkdir(parents=True, exist_ok=True)

N_WORKERS = 30  # number of threads

# -------------------------------------------------------
# WORKER FUNCTION
# -------------------------------------------------------
def load_and_sum(fn):
    """Load a multiband TIF and compute sum-of-bands, ignoring zero pixels."""
    try:
        with rasterio.open(fn) as src:
            arr = src.read().astype(np.float32)

        # Replace sentinel nodata values and 255 as NA
        arr[arr < -1e9] = np.nan
        arr[arr == 255] = np.nan

        # Sum across bands
        arr_sum = np.nansum(arr, axis=0).flatten()

        # Drop NaNs
        clean = arr_sum[~np.isnan(arr_sum)]

        # Ignore zero-valued pixels
        clean = clean[clean != 0]

        if clean.size == 0:
            print(f"[WARNING] All-zero or empty scene: {fn}")
            return None, None

        # Compute scene statistics
        stats = {
            "scene": fn,
            "min": float(np.nanmin(clean)),
            "max": float(np.nanmax(clean)),
            "mean": float(np.nanmean(clean)),
            "std": float(np.nanstd(clean)),
            "n_pixels": int(clean.size)
        }

        return stats, clean

    except Exception as e:
        print(f"[ERROR] {fn}: {e}")
        return None, None

# -------------------------------------------------------
# SCAN ALL VRT FILES
# -------------------------------------------------------
search_path = Path(cube_dir) 
files = sorted(search_path.rglob("*.vrt"))

all_vals = []
scene_stats = []

# -------------------------------------------------------
# PARALLEL PROCESSING USING THREADS
# -------------------------------------------------------
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(load_and_sum, fn): fn for fn in files}

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
        stats, clean = fut.result()

        if stats is None:
            continue

        scene_stats.append(stats)
        all_vals.append(clean)

        fn = stats["scene"]

        # PLOT PER-SCENE HISTOGRAM
        plt.figure(figsize=(8,5))
        plt.hist(clean, bins=80)
        plt.title(f"Sum-of-bands histogram\n{os.path.basename(fn)}")
        plt.xlabel("Sum of bands")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"{Path(fn).stem}_sum_hist.png")
        plt.close()

# -------------------------------------------------------
# GLOBAL HISTOGRAM
# -------------------------------------------------------
if len(all_vals) == 0:
    print("[WARNING] No valid pixel values found across scenes; skipping global histogram and overall stats.")
else:
    all_vals = np.concatenate(all_vals)

    plt.figure(figsize=(10,6))
    plt.hist(all_vals, bins=100)
    plt.title("GLOBAL Histogram — Sum of Bands (All Scenes)")
    plt.xlabel("Sum of bands")
    plt.ylabel("Frequency")
    plt.tight_layout()

    hist_path = output_dir / "GLOBAL_sum_hist.png"
    plt.savefig(hist_path)
    plt.close()

    # also copy global histogram to parent (cube_dir)
    parent_dir = Path(cube_dir)
    try:
        shutil.copy2(hist_path, parent_dir / hist_path.name)
    except Exception as e:
        print(f"[WARNING] could not copy global histogram to parent dir: {e}")

    # -------------------------------------------------------
    # SAVE SCENE STATISTICS
    # -------------------------------------------------------
    df_stats = pd.DataFrame(scene_stats)

    # Try to extract a date from the scene filename (supports YYYYMMDD, YYYY-MM-DD or YYYY_MM_DD)
    date_regex = re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2}|\d{8})')

    def extract_date_from_path(p):
        m = date_regex.search(str(p))
        if not m:
            return pd.NaT
        s = m.group(0)
        s = s.replace('_', '').replace('-', '')
        try:
            return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
        except Exception:
            return pd.to_datetime(s, errors='coerce')

    df_stats['date'] = df_stats['scene'].apply(extract_date_from_path)

    # Save CSV only to output_dir
    csv_path = output_dir / "GLOBAL_statistics.csv"
    try:
        df_stats.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[ERROR] saving statistics: {e}")

    # Also copy CSV to parent directory
    try:
        shutil.copy2(csv_path, parent_dir / csv_path.name)
    except Exception as e:
        print(f"[WARNING] could not copy stats file to parent dir: {e}")

    # -------------------------------------------------------
    # PLOT MEAN vs DATE
    # -------------------------------------------------------
    valid = df_stats.dropna(subset=['date']).copy()
    if not valid.empty:
        valid.sort_values('date', inplace=True)
        plt.figure(figsize=(12,6))
        
        # Plot the per-date mean
        plt.plot(valid['date'], valid['mean'], marker='o', linestyle='-')
        
        # Add horizontal line at the mean across all dates
        global_mean = valid['mean'].mean()
        plt.axhline(y=global_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Global Mean: {global_mean:.2f}')
        
        plt.title('Mean Sum-of-bands per Scene over Time')
        plt.xlabel('Date')
        plt.ylabel('Mean sum of bands')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        ts_path = output_dir / 'mean_by_date.png'
        plt.savefig(ts_path)
        plt.close()

        # copy to parent
        try:
            shutil.copy2(ts_path, parent_dir / ts_path.name)
        except Exception as e:
            print(f"[WARNING] could not copy time-series plot to parent dir: {e}")
    else:
        print("[WARNING] No valid dates extracted; skipping time-series plot.")

    print("\nProcessing complete.")
    print(f"Plots saved to: {output_dir} and {parent_dir}")
    print(f"Stats saved to: {csv_path} and {parent_dir / csv_path.name}")
