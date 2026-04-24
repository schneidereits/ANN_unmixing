# ANN_unmixing: Spectral Unmixing with Artificial Neural Networks

## Overview

This repository implements an end-to-end workflow for **spectral unmixing of hyperspectral imagery** using artificial neural networks (ANNs). Given a spectral library of known land cover types, this system trains a neural network to predict fractional abundances (cover fractions) of each component in mixed pixels from hyperspectral data.

**Key capability**: Transforms multi-class spectral libraries into trained ANN models that can predict vegetation fractions, soil coverage, and other land cover components directly from hyperspectral sensor data (e.g., ENMAP satellites).

### Why Spectral Unmixing?
Hyperspectral pixels often represent mixed land cover targets. This workflow unmixes those pixels into component fractions, enabling detailed land cover classification and monitoring at sub-pixel resolution.

## Getting Demo Data

Demo datasets are available for quick testing and demonstration of the full workflow.

**Where to get data**: https://box.hu-berlin.de/d/648c69949afd48d9b635/

**Contact for access**: schnesha@hu-berlin.de

**Setup instructions**:
1. Download and unzip the demo data folder
2. Place the unzipped folder into your local repository (in the `data/` directory)
3. The preconfigured file paths in the demo `prm_*.py` files will then be valid and ready to use

# Workflow Overview ##################################################################################################################################

The workflow consists of five main phases:

### 1. Data Preparation and Endmember Generation
- **Endmember Selection**: Extracts representative spectral signatures from various land cover classes
- **Filtering**: Applies filters to remove bad wavelengths and improve signal quality
IMPORTANT FILTERING PARAMETERS ARE SET IN THE PRM

- **Library Organization**: Organizes endmembers into different categories (CHAR, fire libraries, plant life forms)

### 2. Synthetic Mixtures Generation
- **Training Data Creation**: Generates synthetic mixtures by combining endmember spectra
- **Fraction Labeling**: Creates corresponding fractional abundance labels for training

### 3. Neural Network Training
- **Model Training**: Trains a neural network to predict vegetation fractions from hyperspectral data

### 4. Prediction Processing
- **Parallel Prediction**: Processes full ENMAP datasets in parallel across multiple tiles
- **Result Storage**: Stores individual tile predictions in structured directories

### 5. Mosaic and Visualization
- **Mosaic Creation**: Combines individual tile predictions into virtual mosaics (VRT files)
- **Statistical Analysis**: Computes statistics on prediction results
- **Time Series Visualization**: Generates time series graphs and animations

## Key Components

### Scripts

#### Training Phase Scripts:
- `99_train.py`: Main training orchestration script
- `scripts/01_endmember_filter_and_format.py`: Endmember filtering and formatting
- `scripts/02_synthmix.py`: Synthetic mixture generation
- `scripts/03_model_train.py`: Neural network model training

#### Prediction Phase Scripts:
- `99_predict.py`: Main prediction orchestration script  
- `scripts/04_predict_parallel.py`: Parallel prediction processing across tiles
- `scripts/05_mosaic_frac.py`: Creates virtual mosaics from tile predictions
- `scripts/05_predict_stats.py`: Computes statistical summaries of predictions


### Parameter Files
- `prm_*.py`: Configuration files for different workflow variants

## Usage Instructions

### Step 1: Configure Your Workflow
1. **Create or select a parameter file** in the `prm/` directory:
   - Copy an existing `prm_demo_*.py` and rename it for your project
   - Or use one of the demo files directly for testing

2. **Set key parameters** in your chosen `prm_*.py` file:
   - `SPECTRAL_LIB`: Path to your spectral library CSV file (see External Data section)
   - `OUTPUT_BASE_LOCATION`: Where to save results (defaults to `output/` directory)
   - `CLASSES`: List of class names in your library (e.g., `['GV', 'NPV', 'Soil']`)
   - `CLASS_COL`: Name of the column in your CSV that identifies classes
   - `FILTER_ENDMEMBERS`: Whether to apply quality filters (set `FILTER_ENDMEMBERS = False` to skip)
   - `BAD_WAVELENGTHS_CSV`: Optional file listing wavelengths to exclude during processing

### Step 2: Edit the Main Training Script
In `99_train.py`, uncomment/select your parameter file:
```python
PRM_MODULE = "prm_demo_PLF_STM"  # Change this to your prm file name
```

### Step 3: Run Training Pipeline
```bash
python 99_train.py
```

This orchestrates the following phases in sequence:
1. **Endmember filtering & formatting** – Cleans and formats spectral signatures from your library
2. **Synthetic mixture generation** – Creates synthetic training data by blending endmembers
3. **Neural network training** – Trains the ANN model on synthetic mixtures
4. **Model export** – Saves the trained model for predictions

Results are saved in `output/<your_project_name>/` with subdirectories for each phase.

### Step 4: Run Predictions on ENMAP Data
Once training completes, create a similar parameter file or use an existing one, then:

1. **Edit the prediction script** in `99_predict.py`:
```python
PRM_MODULE = "prm_demo_PLF_STM"  # Match your prm file
```

2. **Run predictions**:
```bash
python 99_predict.py
```

This will:
1. **Load the trained model** from the training output
2. **Process ENMAP data tiles in parallel** to generate fraction predictions
3. **Create virtual mosaics (VRT)** combining all tiles
4. **Compute statistics** on the predictions
5. **Generate visualizations** and optional time-series animations

Results are saved in `output/<your_project_name>/04_predictions/` with one subdirectory per tile.

## External Data Import

This workflow requires three types of external data inputs:

### 1. Spectral Library (Required)
A CSV file containing spectral reflectance signatures of known land cover classes.

**Format**:
- First column: `class` (or custom name via `CLASS_COL` parameter) – Land cover category names
- Subsequent columns: Numeric wavelengths (e.g., `404`, `409.5`, `415`) representing reflectance at each band
- Rows: Individual spectral measurements (multiple observations per class are averaged)

**Example structure**:
```
class,404,409.5,415,...,2400
GV,0.045,0.048,0.050,...,0.180
GV,0.042,0.050,0.052,...,0.185
NPV,0.055,0.060,0.065,...,0.250
Soil,0.065,0.070,0.075,...,0.290
```

**Create/obtain libraries by**:
- Field sampling campaigns with spectral radiometers
- Existing lab-measured spectral libraries (USGS, EcoSIS, ASTER)
- Hyperspectral image extraction from reference sites
- Laboratory reflectance spectroscopy measurements

**Where to place**: Set the full path in your `prm_*.py` file under `SPECTRAL_LIB`

### 2. ENMAP Data Cubes (Required for Prediction Only)
Hyperspectral image data from ENMAP or compatible sensors to unmix.

**Expected format**:
- Organized in tile subdirectories (e.g., `X0004_Y0014/`) under `prm['DATA_DIR']`
- Each tile contains GeoTIFF or HDF files with all spectral bands
- Coordinate reference system and geolocation metadata required

**Folder structure expected**:
```
data_cube/
├── X0004_Y0014/
│   ├── ENMAP*.tif 
├── X0004_Y0015/
├── X0005_Y0014/
└── X0005_Y0015/
```

**Set path in**: `prm['DATA_DIR']` parameter

### 3. Bad Wavelengths List (Optional)
CSV file listing wavelengths to exclude (e.g., bands with atmospheric absorption or noise).

**Format**:
- Single column of numeric values representing wavelengths to remove
- No header row

**Example**:
```
940
1120
1400
1900
2500
```

**Why exclude bands**:
- Water vapor and CO₂ absorption bands in atmosphere (around 940, 1120, 1400, 1900 nm)
- Instrument noise or dead detectors
- Region outside sensor sensitivity

**Set path in**: `BAD_WAVELENGTHS_CSV` parameter (leave as `None` to disable filtering)

**Default file**: `auxiliary/bad_wavelengths.csv` (configured for ENMAP)

### 4. Reference Data (Optional - for Validation)
Georeferenced shapefiles or geopackages containing ground truth labels for validation.

**Format**: GeoPackage (.gpkg) or Shapefile (.shp) with class polygons
**Used in**: Optional validation scripts (in demo folders)
**Example files**: `reference_fractions.gpkg`, `reference_fcov_taylor_mountain.gpkg`

## Dependencies

### Core Python Libraries:
- Python 3.x
- TensorFlow/Keras (for neural network implementation)
- NumPy (for numerical operations)
- Pandas (for data manipulation)
- SciPy (for scientific computing)
- Matplotlib (for visualization)
- Seaborn (for advanced plotting)
- tqdm (for progress bars)
- GDAL/OGR (for geospatial data handling)
- scikit-learn (for machine learning utilities)
- Concurrent futures (Python standard library for parallel processing)
- Multiprocessing (Python standard library for parallel processing)

## Repository Structure

```
.
├── .git/
├── .gitignore
├── 99_predict.py
├── 99_train.py
├── auxiliary/
│   ├── all_wavelengths.csv
│   └── bad_wavelengths.csv
├── data/
│   ├── auxiliary/
│   │   └── roi.gpkg
│   └── data_cube/
├── LICENSE
├── prm/
│   └── prm_demo.py
├── scripts/
│   ├── 00_library_plots.ipynb
│   ├── 00_sample_endmember_spectra.py
│   ├── 01_endmember_filter_and_format.py
│   ├── 02_synthmix.py
│   ├── 03_model_train.py
│   ├── 03_model_train_old.py
│   ├── 04_predict_parallel.py
│   ├── 05_mosaic_frac.py
│   ├── 05_mosaic_frac_time_series.py
│   ├── 05_predict_stats.py
│   └── 05_visualization/
│       ├── time_series_graph/
│       │   ├── 99_sampling_values_library_multiprocessing.py
│       │   └── 99_plot_time_series_line.py
│       └── time_series_animation/
│           └── 02_enmapcube_visualization_and_animation_fcover_image_cc.py
└── README.md

