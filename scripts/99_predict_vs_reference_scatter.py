import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import os
gdal.UseExceptions()
# ref data 
gpkg_path = r"D:\ANN_unmixing\data\demodata_taylorM\reference_fcov_taylor_mountain.gpkg"

# File paths for predicted data and output plot
prm = "demo_taylor_Mt_with_eco"  # Set this to the name of your prm file (without .py extension)
# Using os.path.join for cross-platform compatibility
tif_path = f"D:/ANN_unmixing/output/{prm}/04_predictions/enmapl2a_20250729_VEGCOV_FRAC.TIF"
plot_output_path = f"D:/ANN_unmixing/output/{prm}/05_visualization/scatter_plots.png"


# Load GeoPackage file
print("Loading GeoPackage file...")
gdf = gpd.read_file(gpkg_path)
print(f"GeoPackage loaded successfully. Shape: {gdf.shape}")
print(f"Columns in GeoPackage: {list(gdf.columns)}")

# Load TIFF file
print("\nLoading TIFF file...")
dataset = gdal.Open(tif_path)
if dataset is None:
    raise Exception("Failed to open TIFF file")

# Get image dimensions and number of bands
width = dataset.RasterXSize
height = dataset.RasterYSize
num_bands = dataset.RasterCount
print(f"TIFF dimensions: {width} x {height}, Bands: {num_bands}")

# Extract band names from metadata
band_names = []
for i in range(num_bands):
    band = dataset.GetRasterBand(i + 1)
    
    # Try to get band description or name
    band_desc = band.GetDescription()
    if band_desc:
        band_names.append(band_desc + "_predicted")
    else:
        # Try to get unit or color interpretation
        unit = band.GetUnitType()
        if unit:
            band_names.append(f"band_{i+1}_{unit}")
        else:
            # Fall back to generic naming
            band_names.append(f"band_{i+1}")

print(f"Extracted band names: {band_names}")

# Extract all band values for each point
print("\nExtracting band values...")
band_values = []
for i in range(num_bands):
    band = dataset.GetRasterBand(i + 1)
    band_array = band.ReadAsArray()
    band_values.append(band_array)

# Get coordinates from GeoPackage
points = gdf.geometry
x_coords = [point.x for point in points]
y_coords = [point.y for point in points]

# Sample the TIFF data at each point location
sampled_values = []
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    # Convert geographic coordinates to pixel coordinates
    # Get geotransform information
    transform = dataset.GetGeoTransform()
    x_pixel = int((x - transform[0]) / transform[1])
    y_pixel = int((y - transform[3]) / transform[5])
    
    # Check if pixel coordinates are within bounds
    if 0 <= x_pixel < width and 0 <= y_pixel < height:
        # Extract values from all bands at this pixel
        band_data = []
        for j in range(num_bands):
            val = band_values[j][y_pixel, x_pixel]
            band_data.append(val)
        sampled_values.append(band_data)
    else:
        # If out of bounds, append NaN values
        sampled_values.append([np.nan] * num_bands)
        
# Close the TIFF dataset to free up resources
dataset = None

# Create DataFrame with actual band values
df_bands = pd.DataFrame(sampled_values, columns=band_names)

# Merge with original GeoPackage data
df_merged = pd.concat([gdf.reset_index(drop=True), df_bands], axis=1)
print("\nMerged DataFrame created successfully!")
print(f"Shape: {df_merged.shape}")
print(f"Column names: {list(df_merged.columns)}")

# Print first few rows of the merged dataframe
print("\nFirst 5 rows of merged dataframe:")
print(df_merged.head())



# Assuming 'df_merged' is your DataFrame with the data
# Create subplots for the three scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: GV vs GV Predicted
axes[0].scatter(df_merged['FCOV_GV'], df_merged['GV_predicted'], alpha=0.7)
axes[0].plot([0, 100], [0, 100], 'r--', lw=2)  # Diagonal line
axes[0].set_xlim(0, 100)
axes[0].set_ylim(0, 100)
axes[0].set_xlabel('FCOV_GV')
axes[0].set_ylabel('GV Predicted')

# Calculate R² and MAE for GV vs GV Predicted
r2_gv = r2_score(df_merged['FCOV_GV'], df_merged['GV_predicted'])
mae_gv = mean_absolute_error(df_merged['FCOV_GV'], df_merged['GV_predicted'])
axes[0].text(0.05, 0.95, f'R² = {r2_gv:.3f}\nMAE = {mae_gv:.3f}', 
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: NPV vs NPV Predicted
axes[1].scatter(df_merged['FCOV_NPV'], df_merged['NPV_predicted'], alpha=0.7)
axes[1].plot([0, 100], [0, 100], 'r--', lw=2)  # Diagonal line
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)
axes[1].set_xlabel('FCOV_NPV')
axes[1].set_ylabel('NPV Predicted')

# Calculate R² and MAE for NPV vs NPV Predicted
r2_npv = r2_score(df_merged['FCOV_NPV'], df_merged['NPV_predicted'])
mae_npv = mean_absolute_error(df_merged['FCOV_NPV'], df_merged['NPV_predicted'])
axes[1].text(0.05, 0.95, f'R² = {r2_npv:.3f}\nMAE = {mae_npv:.3f}', 
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: SUB vs NV Predicted
axes[2].scatter(df_merged['FCOV_SUB'], df_merged['NV_predicted'], alpha=0.7)
axes[2].plot([0, 100], [0, 100], 'r--', lw=2)  # Diagonal line
axes[2].set_xlim(0, 100)
axes[2].set_ylim(0, 100)
axes[2].set_xlabel('FCOV_SUB')
axes[2].set_ylabel('NV Predicted')

# Calculate R² and MAE for SUB vs NV Predicted
r2_sub = r2_score(df_merged['FCOV_SUB'], df_merged['NV_predicted'])
mae_sub = mean_absolute_error(df_merged['FCOV_SUB'], df_merged['NV_predicted'])
axes[2].text(0.05, 0.95, f'R² = {r2_sub:.3f}\nMAE = {mae_sub:.3f}', 
             transform=axes[2].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
plt.show()