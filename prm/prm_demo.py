import math
import os
import pandas as pd

# -----------------------
# 00 Path managment based on prm file name and location 
# -----------------------

# Get the current file name without extension
prm_name = os.path.splitext(os.path.basename(__file__))[0]

# Path to the prm file
script_dir = os.path.dirname(os.path.abspath(__file__))
prm_file = os.path.join(script_dir, f"{prm_name}.py")

if not os.path.exists(prm_file):
    raise FileNotFoundError(f"Could not find prm file: {prm_file}")

# Predefine variables expected by the prm module
BASE_DIR = os.path.abspath(os.path.dirname(prm_file))

# Set NAME to suffix after 'prm_' or default to the full file name
if prm_name.startswith("prm_"):
    NAME = prm_name.split("prm_", 1)[1]
else:
    NAME = prm_name
    
# Central output root (all produced outputs go here)
OUTPUT_ROOT = os.path.join(BASE_DIR, NAME)
# create output root if it does not exist when params are imported
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------
# input data and directory paths 
# -----------------------

# Input data files 
REFLECTANCE_FILE = r"E:\Project_EnFireMap\01_data\03_spectral_libraries\temp_shawn\resampling\spectral_library_resampled_enmap_interpolated.csv"
BAD_WAVELENGTHS_CSV = r"E:\Project_EnFireMap\01_data\03_spectral_libraries\temp_shawn\resampling\wavelength\bad_wavelengths.csv"

# -----------------------
# Out dir creation
# -----------------------

# Directory layout (project-relative by default)
ENDMEMBER_DIR = os.path.join(OUTPUT_ROOT, '01_endmembers')
DATA_DIR = ENDMEMBER_DIR
SYNTHMIX_DIR = os.path.join(OUTPUT_ROOT, '02_synthmix')
# Place output directories under OUTPUT_ROOT to keep all generated products together
MODEL_DIR = os.path.join(OUTPUT_ROOT, '03_model')
PREDICTIONS_DIR = os.path.join(OUTPUT_ROOT, '04_predictions')
VISUALIZATION_DIR = os.path.join(OUTPUT_ROOT, '05_visualization')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

# ensure output subfolders exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


# -----------------------
# 01 endmember filtering
# -----------------------
# Define a function to apply the filtering logic to the endmember DataFrame

def filter_endmembers(df: pd.DataFrame, BAND_MAP: dict, numeric_cols: list) -> pd.DataFrame:
    """
    Apply filtering using temporary NDVI and CAI for conditions.
    NDVI and CAI are not added to df, so they won't be written to CSVs.
    """
    before_count = len(df)

    # Compute temporary NDVI and CAI
    NDVI = (df[BAND_MAP['800']] - df[BAND_MAP['670']]) / (df[BAND_MAP['800']] + df[BAND_MAP['670']])
    CAI  = 0.5 * (df[BAND_MAP['2000']] + df[BAND_MAP['2200']]) - df[BAND_MAP['2100']]

    # Demo Class-based filters to remove outliers 
    df = df[~((df['class'] == 'GV') & (NDVI < 0.8))]
    df = df[~((df['class'] == 'NPV') & (CAI > 0.25))]
   
    after_count = len(df)
    print(f"Filtered {before_count - after_count} records. Remaining: {after_count}")

    return df

 
# ----------------------
# 02_synthmix
# ----------------------
CLASSES = ['GV', 'NPV', 'SUB', 'CHAR']
SYNTHMIX_INPUT_FILES = [f"{c}.csv" for c in CLASSES]  
SYNTHMIX_OUTPUT_SPEC = 'mixed_spectra.npy'
SYNTHMIX_OUTPUT_FRAC = 'fraction_label.npy'
ENDMEMBER_CSV = None
CLASS_COL = 'class'
input_file_names = [f"{c}.csv" for c in CLASSES]
output_file_name_spec = SYNTHMIX_OUTPUT_SPEC
output_file_name_frac = SYNTHMIX_OUTPUT_FRAC

# Synthmix parameters
NUMBER_OF_SAMPLES = 256000
EQUALIZE_SAMPLES = False
if EQUALIZE_SAMPLES:
    NUMBER_OF_SAMPLES = NUMBER_OF_SAMPLES*1
CLASS_PROBABILITIES = None
MIXING_COMPLEXITY_PROBABILITIES = [0.1, 0.4, 0.3, 0.1, 0.1]
INCL_PURE_LIBRARY = False

# ----------------------
# 03_model_train
# ----------------------
# Model parameters
# Adpated from Klehr et al. 2025, with adjustment for  MODEL_DENSE_UNITS and EPOCHS
# https://doi.org/10.1016/j.rse.2025.114740
# -----------------------
MODEL_INPUT_SHAPE = (204,)
MODEL_DENSE_UNITS = 256
MODEL_N_LAYERS = 5
MODEL_NUM_CLASSES = len(CLASSES)

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
LEARNING_DECAY_RATE = 0.5

# Misc
RANDOM_SEED = 42
USE_GPU = False

# Backwards-compatible local names for scripts that expect them
WORK_DIR = BASE_DIR
FOLDER_INPUT = SYNTHMIX_DIR
FILE_NAME_X_IN = SYNTHMIX_OUTPUT_SPEC
FILE_NAME_Y_IN = SYNTHMIX_OUTPUT_FRAC
FOLDER_OUTPUT = MODEL_DIR
FILE_NAME_MODEL = 'nn_model'

# ----------------------
# 04_predict_parallel (parameters for prediction workflows)
# ----------------------
N_WORKERS = math.ceil(os.cpu_count() * 0.7)
PARALLELISM_THREADS = 1  # threads per process (workers * threads = total cores)

CUBE_SPEC = r"/data/Dagobah/enmap/dc_cali/enmap/03_EnMAP_cube/cube_v2_red"
CUBE_FRAC = PREDICTIONS_DIR

REG_MODEL_PATH = os.path.join(MODEL_DIR, 'nn_model.keras')
FN_LOG_FILE = os.path.join(CUBE_FRAC, 'processing_log.csv')

CUBE_AUX_MASKS = r"/data/Dagobah/enmap/dc_cali/enmap/04_aux_data"
AUX_MASK_FILENAMES = ['LND_2023-2024_MASK_WATER.TIF']

CLASS_NAMES = CLASSES
APPLY_CLIP = True
APPLY_MASK = True
APPLY_AUX_MASKS = True

IGNORE_HAZE = False
# Quality sub-masks used when IGNORE_HAZE = True
QUAL_SUBMASKS = [
    'QL_QUALITY_CIRRUS.TIF',
    'QL_QUALITY_CLOUD.TIF',
    'QL_QUALITY_CLOUDSHADOW.TIF',
    'QL_QUALITY_SNOW.TIF'
]


# Tiles to process (single authoritative list)
TILES_TO_PROCESS = ["X0012_Y0028", 
                    "X0012_Y0029",
                    "X0012_Y0030",
                    "X0013_Y0028", 
                    "X0013_Y0029",
                    "X0013_Y0030"]

# ----------------------
# 05_mosaic_frac
# ----------------------
# default product directory and file products
PRODUCTS = ['VEGCOV_FRAC.TIF']


# End of params-only configuration file

