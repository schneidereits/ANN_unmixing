import os
import re
import numpy as np
import tensorflow as tf
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import sys
import importlib.util
################################################################################
#                                  User Settings                               #
################################################################################

# Read configuration from central params file (prm.py)

# Get the module name from the environment
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

n_workers = prm.N_WORKERS
parallelism_threads = prm.PARALLELISM_THREADS
cube_spec = prm.CUBE_SPEC
cube_frac = prm.CUBE_FRAC
fn_reg_model = prm.REG_MODEL_PATH
fn_log_file = prm.FN_LOG_FILE
cube_aux_masks = prm.CUBE_AUX_MASKS
aux_mask_filenames = prm.AUX_MASK_FILENAMES
class_names = prm.CLASS_NAMES
apply_clip = prm.APPLY_CLIP
apply_mask = prm.APPLY_MASK
apply_aux_masks = prm.APPLY_AUX_MASKS
tiles_to_process = prm.TILES_TO_PROCESS
ignore_haze = prm.IGNORE_HAZE
QUAL_SUBMASKS = prm.QUAL_SUBMASKS

################################################################################
#                           Function Definitions                               #
################################################################################

def norm(a):
    a_norm = a.astype(np.float32)
    return a_norm / 10000.


def toRaster(fraction, geotransform, projection, fn_frac_img, class_names):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = fraction.shape[0:2]
    n = fraction.shape[-1]

    os.makedirs(os.path.dirname(fn_frac_img), exist_ok=True)
    outdata = driver.Create(fn_frac_img, cols, rows, n, gdal.GDT_Byte)
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)

    for i in range(n):
        out_band = outdata.GetRasterBand(i + 1)
        out_band.WriteArray(fraction[..., i])
        out_band.SetNoDataValue(-1)
        out_band.SetDescription(class_names[i])

    outdata.FlushCache()
    outdata = None


def predict_tile(task):
    """Process a single tile: load model, run prediction, save output."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(parallelism_threads)
    tf.config.threading.set_inter_op_parallelism_threads(parallelism_threads)
    
    (fn_spec_img, fn_mask_noda, qual_mask_files, fn_frac_img,
     fn_reg_model, class_names, apply_clip, apply_mask,
     apply_aux_masks, aux_mask_files,ignore_haze) = task

    try:
        ds = gdal.Open(fn_spec_img)
        x = ds.ReadAsArray()
        x = np.moveaxis(x, 0, -1)
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()

        # --- Always apply no-data mask ---
        mask_noda = gdal.Open(fn_mask_noda).ReadAsArray() == 1
        null_mask = mask_noda.copy()   # initialize with no-data mask

        if apply_mask & ignore_haze == False:
            mask_qual = gdal.Open(qual_mask_files[0]).ReadAsArray() == 1
            null_mask = np.logical_or(null_mask, mask_qual)
            
        if apply_mask & ignore_haze:
            for qual_mask_file in qual_mask_files:
                mask_qual = gdal.Open(qual_mask_file).ReadAsArray() >= 1
                null_mask = np.logical_or(null_mask, mask_qual)


        if apply_aux_masks and aux_mask_files:
            for aux_mask_file in aux_mask_files:
                if os.path.exists(aux_mask_file):
                    aux_mask = gdal.Open(aux_mask_file).ReadAsArray() == 1
                    null_mask = np.logical_or(null_mask, aux_mask)

        model = tf.keras.models.load_model(fn_reg_model, compile=False)
        num_classes = len(class_names)
        x_out = np.zeros((x.shape[0], x.shape[1], num_classes), dtype=np.float32)

        # Predict per row
        for i in range(x.shape[0]):
            x_pred = model(norm(x[i, ...]), training=False).numpy()
            # Ensure output is bounded between 0 and 1 before applying scaling
            x_pred = np.clip(x_pred, 0, 1)
            # Apply softmax to ensure outputs sum to 1
            #x_pred = tf.nn.softmax(x_pred, axis=-1).numpy()
            x_out[i, ...] = x_pred

        x_out = (x_out * 100).astype(np.int8)
        if apply_clip:
            x_out[x_out > 100] = 100
            x_out[x_out < 0] = 0
        # apply mask
        x_out[null_mask] = -1
        
        # Close mask datasets
        null_mask = None
        mask_qual = None
        aux_mask  = None

        toRaster(x_out, geotransform, projection, fn_frac_img, class_names)
        ds = None
        return fn_spec_img, "DONE"

    except Exception as e:
        return fn_spec_img, f"ERROR: {e}"


################################################################################
#                               Execution                                      #
################################################################################
def main():
    """
    Process all spectral tiles in parallel and save results to log file.
    """
    print('\n=== Script Execution Started ===')

    if os.path.exists(fn_log_file):
        df_log = pd.read_csv(fn_log_file)
    else:
        df_log = pd.DataFrame(columns=['Image Name', 'Status'])

    subfolder_pattern = r"X\d{4}_Y\d{4}"
    tasks = []

    for root, dirs, files in os.walk(cube_spec):
        tiles_pattern = re.compile("|".join(tiles_to_process))
        if not tiles_pattern.search(os.path.basename(root)):
            continue

        for file in files:
            if not file.endswith('SPECTRAL_IMAGE.TIF'):
                continue

            fn_spec_img = os.path.join(root, file)
            fn_mask_noda = fn_spec_img.replace('SPECTRAL_IMAGE.TIF', 'MASK_NODA.TIF')
            if ignore_haze:
                qual_mask_files = [
                    fn_spec_img.replace('SPECTRAL_IMAGE.TIF', m)
                    for m in QUAL_SUBMASKS
                ]
            else:
                qual_mask_files = [
                    fn_spec_img.replace('SPECTRAL_IMAGE.TIF', 'MASK_QUAL.TIF')
                ]

            fn_frac_img = os.path.join(cube_frac, os.path.relpath(root, cube_spec),
                                       file.replace('SPECTRAL_IMAGE.TIF', 'VEGCOV_FRAC.TIF'))
            aux_mask_files = [os.path.join(cube_aux_masks, os.path.relpath(root, cube_spec), aux)
                              for aux in aux_mask_filenames] if aux_mask_filenames else []

            if any(df_log['Image Name'] == fn_spec_img) and \
                    df_log[df_log['Image Name'] == fn_spec_img]['Status'].iloc[0] == 'DONE':
                continue

            tasks.append((fn_spec_img, fn_mask_noda, qual_mask_files, fn_frac_img,
                          fn_reg_model, class_names, apply_clip, apply_mask,
                          apply_aux_masks, aux_mask_files,ignore_haze))

    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(predict_tile, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
            results.append(f.result())

    # Logging
    for fn, status in results:
        df_log = pd.concat([df_log, pd.DataFrame({'Image Name': [fn], 'Status': [status]})],
                           ignore_index=True)

    df_log.to_csv(fn_log_file, index=False)
    print('\n=== Script Execution Completed ===')


if __name__ == '__main__':
    main()

