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

# -----------------------
# OUTPUT LOCATION MANAGEMENT
# -----------------------
# CONFIGURABLE: Set your desired output base location here
# Options:
#   - Use absolute path: r"C:\path\to\outputs"
#   - Use relative to project root: "output"
#   - Use prm directory: None (will default to prm folder)
OUTPUT_BASE_LOCATION = "output"  # Set to None to use prm directory

# Resolve output base location
if OUTPUT_BASE_LOCATION is None:
    OUTPUT_BASE_LOCATION = BASE_DIR  # Default to prm directory
elif not os.path.isabs(OUTPUT_BASE_LOCATION):
    # Make relative paths relative to project root (parent of prm dir)
    OUTPUT_BASE_LOCATION = os.path.join(os.path.dirname(BASE_DIR), OUTPUT_BASE_LOCATION)

OUTPUT_BASE_LOCATION = os.path.abspath(OUTPUT_BASE_LOCATION)

# Central output root (all produced outputs go here)
OUTPUT_ROOT = os.path.join(OUTPUT_BASE_LOCATION, NAME)
# create output root if it does not exist when params are imported
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------
# input data and directory paths 
# -----------------------
STM = True
# STM-specific parameters (when STM=True)
STM_METRICS = ['p10', 'p25', 'p50', 'p75', 'p90']  # Percentiles/metrics for each wavelength band
STM_N_BAND_PER_METRIC = 204  # Number of unique wavelength bands per metric (204 wavelengths x 5 metrics = 1020 total)

# Input data files 
SPECTRAL_LIB = r"data\endmembers\Plant_life_forms_STMS.csv"

BAD_WAVELENGTHS_CSV = r"auxiliary\bad_wavelengths.csv"
if STM:
    BAD_WAVELENGTHS_CSV = None

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
FILTER_ENDMEMBERS = False  # Set to False to skip filtering and use the full library
# Define a function to apply the filtering logic to the endmember DataFrame

def filter_endmembers(df: pd.DataFrame, BAND_MAP: dict, numeric_cols: list) -> pd.DataFrame:
    """
    Apply filtering using temporary NDVI and CAI for conditions.
    NDVI and CAI are not added to df, so they won't be written to CSVs.
    """
    before_count = len(df)

    # Compute temporary NDVI and CAI
    # Reset index to ensure alignment with df for boolean indexing
    NDVI = (df[BAND_MAP['800']] - df[BAND_MAP['670']]) / (df[BAND_MAP['800']] + df[BAND_MAP['670']])
    NDVI = NDVI.reset_index(drop=True)
    
    CAI  = 0.5 * (df[BAND_MAP['2000']] + df[BAND_MAP['2200']]) - df[BAND_MAP['2100']]
    CAI = CAI.reset_index(drop=True)

    # Demo Class-based filters to remove outliers 
    df = df.reset_index(drop=True)  # Reset df index to match the computed Series
    #df = df[~((df['class'] == 'GV') & (NDVI < 0.8))]
    #df = df[~((df['class'] == 'NPV') & (CAI > 0.25))]
   
    after_count = len(df)
    print(f"Filtered {before_count - after_count} records. Remaining: {after_count}")

    return df

 
# ----------------------
# 02_synthmix
# ----------------------
CLASSES = ["btr", "ntr", "shr", "her", "nve"]
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
# When STM=True: MODEL_INPUT_SHAPE = len(STM_METRICS) * STM_N_BAND_PER_METRIC
if STM:
    MODEL_INPUT_SHAPE = (len(STM_METRICS) * STM_N_BAND_PER_METRIC,)  
    
MODEL_DENSE_UNITS = 8
MODEL_N_LAYERS = 1
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
N_WORKERS = math.ceil(os.cpu_count() * 0.1)
PARALLELISM_THREADS = 4  # threads per process (workers * threads = total cores)

CUBE_SPEC = r"data\data_cube\STM"

REG_MODEL_PATH = os.path.join(MODEL_DIR, 'nn_model.keras')
FN_LOG_FILE = os.path.join(PREDICTIONS_DIR, 'processing_log.csv')

CUBE_AUX_MASKS =None
AUX_MASK_FILENAMES = None

CLASS_NAMES = CLASSES
APPLY_CLIP = False
APPLY_MASK = False
APPLY_AUX_MASKS = False

IGNORE_HAZE = False
# Quality sub-masks used when IGNORE_HAZE = True
QUAL_SUBMASKS = [
    'QL_QUALITY_CIRRUS.TIF',
    'QL_QUALITY_CLOUD.TIF',
    'QL_QUALITY_CLOUDSHADOW.TIF',
    'QL_QUALITY_SNOW.TIF'
]


# Tiles to process (single authoritative list)
TILES_TO_PROCESS = ["X0004_Y0014", 
                    "X0004_Y0015",
                    "X0005_Y0014",
                    "X0005_Y0015"]

# ----------------------
# 05_mosaic_frac
# ----------------------
# default product directory and file products
PRODUCTS = ['PLF_FRAC.TIF']


# End of params-only configuration file

