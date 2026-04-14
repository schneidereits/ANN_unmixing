# ANN_unmixing

## setup
# Spectral Library Workflow for ENMAP Data Analysis

This repository implements a modular workflow for processing ENMAP hyperspectral data using neural network unmixing techniques. The workflow processes spectral libraries to generate cover fraction maps from hyperspectral imagery.


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

#### Visualization Scripts:
- `scripts/05_visualization/time_series_graph/99_sampling_values_library_multiprocessing.py`: Sampling values from predictions
- `scripts/05_visualization/time_series_graph/99_plot_time_series_line.py`: Time series plotting
- `scripts/05_visualization/time_series_animation/02_enmapcube_visualization_and_animation_fcover_image_cc.py`: Animation generation

### Parameter Files
- `prm_*.py`: Configuration files for different workflow variants

### Server Deployment
- **Training**: Executed on Windows server where auxiliary data is hosted (cant be accessed directly from Linux server)
- **Prediction**: Executed on Linux server where data cubes are stored


## Usage

1. Configure parameters in appropriate `prm_*.py` files
2. Run `99_train.py` on Windows server to train the model
3. Run `99_predict.py` on Linux server to process new data
4. Results are stored in respective output directories that automatically matches the name of the prm
- IPython/Jupyter (for notebooks)