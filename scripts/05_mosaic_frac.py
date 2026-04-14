################################################################################
#                                  Import                                      #
################################################################################

import os
from os import mkdir
from os.path import isdir
from osgeo import gdal
from prm import PREDICTIONS_DIR, PRODUCTS


################################################################################
#                                  User Settings                               #
################################################################################

# Working directory and product (now from centralized params)
cube_dir = PREDICTIONS_DIR
products = PRODUCTS

################################################################################
#                           Function Definitions                               #
################################################################################

#  Color codes for warnings
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def create_subfolder(directory, subfolder_name):
    print('\n--> Create subfolder for virtual mosaics')
    subfolder_path = os.path.join(directory, subfolder_name)
    if not isdir(subfolder_path):
        mkdir(subfolder_path)
        print(f'   ... Created subfolder: {subfolder_name}')
    else:
        print(f'{RED}    ... Subfolder already exisits: {subfolder_name}{RESET}')
    return subfolder_path

def get_unique_filenames(directory, extension):
    print('\n--> Find unique filenames')
    all_filenames = []
    for root, dirs, files in os.walk(directory):
        for cur_file in files:
            if cur_file.endswith(extension):
                all_filenames.append(cur_file)
    unique_filenames = set(all_filenames)
    print(f'    ... Found {len(unique_filenames)} unique filenames for product {extension}')
    return unique_filenames

def build_vrt(directory, unique_filenames, mosaic_subfolder):
    print('\n--> Create virtual mosaics')
    original_dir = os.getcwd()
    os.chdir(mosaic_subfolder)
    for cur_filename in unique_filenames:
        mosaic_path = cur_filename[:-4] + '.vrt'
        if not os.path.isfile(mosaic_path):
            print(f'   ... Processing: {cur_filename}')
            relative_paths_to_mosaic = []
            for root, dirs, files in os.walk(directory):
                for cur_file in files:
                    if cur_file.endswith(cur_filename):
                        relative_paths_to_mosaic.append(
                            os.path.relpath(os.path.join(root, cur_file), mosaic_subfolder)
                        )
            gdal.BuildVRT(mosaic_path, relative_paths_to_mosaic)
        else:
            print(f'{RED}    ... VRT for {cur_filename} already exists{RESET}')
    os.chdir(original_dir)

def add_metadata(directory, unique_filenames, mosaic_subfolder):
    print('\n--> Copying metadata to all bands in the VRT')
    for cur_filename in unique_filenames:
        print(f'    ... Processing: {cur_filename}')
        full_path_filenames_to_mosaic = []
        for root, dirs, files in os.walk(directory):
            for cur_file in files:
                if cur_file.endswith(cur_filename):
                    full_path_filenames_to_mosaic.append(os.path.join(root, cur_file))

        # Open the first file and the corresponding VRT file
        raster_meta = gdal.Open(full_path_filenames_to_mosaic[0])
        vrt_meta = gdal.Open(os.path.join(mosaic_subfolder, cur_filename[:-4] + '.vrt'))
        band_count = raster_meta.RasterCount

        # Loop through each band and set metadata
        for band_index in range(1, band_count + 1):
            # Extract band metadata and description from the original raster
            band_in_meta = raster_meta.GetRasterBand(band_index).GetMetadata()
            band_in_name = raster_meta.GetRasterBand(band_index).GetDescription()

            # Get the corresponding band in the VRT and set metadata and description
            band_out_meta = vrt_meta.GetRasterBand(band_index)
            band_out_meta.SetMetadata(band_in_meta)
            band_out_meta.SetDescription(band_in_name)

        # Clean up the metadata objects
        raster_meta = None
        vrt_meta = None

################################################################################
#                               Execution                                      #
################################################################################
def main():
    """
    Process spectral images in a cube directory to create mosaics.
    Steps:
      1. Create a subfolder for mosaics.
      2. Get unique filenames for spectral images based on product type.
      3. Build VRT mosaics for each unique filename.
      4. Add metadata to each VRT file.
    """
    print('\n=== Script Execution Started ===')

    # Step 1: Create subfolder for mosaics
    mosaic_subfolder = create_subfolder(cube_dir, 'mosaic')

    for product in products:

        # Step 2: Get unique filenames for spectral images (based on file extension)
        unique_filenames = get_unique_filenames(cube_dir, product)

        # Step 3: Build VRT mosaics for each unique filename
        build_vrt(cube_dir, unique_filenames, mosaic_subfolder)

        # Step 4: Add metadata to each VRT file
        add_metadata(cube_dir, unique_filenames, mosaic_subfolder)

    print('\n=== Script Execution Completed ===')


if __name__ == "__main__":
    main()