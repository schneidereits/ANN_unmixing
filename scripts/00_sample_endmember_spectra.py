import os
import csv
from osgeo import gdal, ogr
from collections import defaultdict

# --- User-defined parameters ---
cube_dir = r'R:\enmap\dc_cali\enmap\03_EnMAP_cube\cube_v2_stm'
mask_dir = r'R:\enmap\dc_cali\enmap\03_EnMAP_cube\cube_v2_stm'
gpkg_path = r"E:\temp\temp_akpona\05_4shawn\cc_data_4shawn\frac_plf\cc_plf_library_merged.gpkg"
output_csv_path = r"E:\temp\temp_akpona\05_4shawn\cc_data_4shawn\frac_plf\01_library\cc_plf_library_pure_STMS.csv"
user_specified_bands = None  # Set to None to extract all bands
num_bands = 1020


def load_sample_points(gpkg_path):
    ds = ogr.Open(gpkg_path, 0)
    lyr = ds.GetLayer()
    sample_points = [(feat.GetFID(), feat.GetField("input_id"),
                      feat.GetGeometryRef().GetX(), feat.GetGeometryRef().GetY(),
                      feat.GetField("class"),feat.GetField("desc")) for feat in lyr]
    return sample_points


def write_results_to_csv(results, output_csv_path, is_header, num_bands=0):
    with open(output_csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if is_header:
            # Use num_bands to construct the band part of the header
            header = ['FID', 'input_id', 'X Coordinate', 'Y Coordinate', 'class', 'desc', 'Subdir', 'Image Name', 'Date',
                      'NoData Mask Value', 'Quality Mask Value'] + [f'Band {i + 1}' for i in range(num_bands)]
            csvwriter.writerow(header)
        else:
            for result in results:
                csvwriter.writerow(result)


def process_images_and_masks(cube_dir, mask_dir, sample_points, user_specified_bands=None):
    results = []
    image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(cube_dir) for f in filenames if
                   f.endswith('_STMS.vrt')]
    total_images = len(image_files)
    processed_images = 0

    for image_path in image_files:

        ds = gdal.Open(image_path)
        if not ds:
            continue
        gt = ds.GetGeoTransform()

        # Extract subdir and date from image_path
        subdir = os.path.basename(os.path.dirname(image_path))
        image_name = os.path.basename(image_path)
        base_name = image_name.replace('_STMS.vrt', '')
        date = image_name.split('_')[1]  # Assuming date is the second element after splitting by '_'

        # Construct paths for NoData and Quality masks
        mask_nodata_path = os.path.join(mask_dir, subdir, f"{base_name}_MASK_NODA.TIF")
        mask_qual_path = os.path.join(mask_dir, subdir, f"{base_name}_MASK_QUAL.TIF")
        ds_nodata = gdal.Open(mask_nodata_path)
        ds_qual = gdal.Open(mask_qual_path)

        for sample_point in sample_points:
            fid, input_id, x, y, cover_class, desc = sample_point
            px = int((x - gt[0]) / gt[1])
            py = int((y - gt[3]) / gt[5])

            if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                band_indices = range(1, ds.RasterCount + 1) if user_specified_bands is None else [
                    int(band.split(' ')[1]) for band in user_specified_bands]
                band_values = [ds.GetRasterBand(band_idx).ReadAsArray(px, py, 1, 1)[0][0] for band_idx in band_indices]

                # Extract NoData and Quality mask values
                nodata_value = ds_nodata.GetRasterBand(1).ReadAsArray(px, py, 1, 1)[0][0] if ds_nodata else None
                quality_value = ds_qual.GetRasterBand(1).ReadAsArray(px, py, 1, 1)[0][0] if ds_qual else None

                results.append(
                    [fid, input_id, x, y, cover_class, desc, subdir, image_name, date, nodata_value, quality_value] + band_values)

        processed_images += 1
        if processed_images % 5 == 0 or processed_images == total_images:
            write_results_to_csv(results, output_csv_path, False)  # False indicates not writing the header
            results = []  # Clear results for the next batch
        print(
            f"Processed {processed_images}/{total_images} images ({round((processed_images / total_images) * 100, 2)}%)")


if __name__ == '__main__':
    sample_points = load_sample_points(gpkg_path)

    # Check if the CSV file already exists to avoid overwriting the header
    file_exists = os.path.exists(output_csv_path)
    if not file_exists:
        write_results_to_csv([], output_csv_path, True, num_bands)  # Now passing num_bands

    process_images_and_masks(cube_dir, mask_dir, sample_points, user_specified_bands)
    print("Processing complete.")
    
    # Create a new CSV file for each unique class containing only band-related columns

    # Read the main CSV file and group rows by class
    class_rows = defaultdict(list)
    with open(output_csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Read the header

        # Identify band-related columns (columns starting with 'Band ')
        band_columns = [i for i, col in enumerate(header) if col.startswith('Band ')]

        for row in csvreader:
            class_rows[row[header.index('class')]].append([row[i] for i in band_columns])  # Group rows by the 'class' column (index 4)

    # Write each class's rows to a separate CSV file
    for cover_class, rows in class_rows.items():
        class_csv_path = os.path.join(os.path.dirname(output_csv_path), f"{cover_class}.csv")
        with open(class_csv_path, 'w', newline='') as class_csvfile:
            csvwriter = csv.writer(class_csvfile)
            # Write only the band-related column headers
            #csvwriter.writerow([header[i] for i in band_columns])
            csvwriter.writerows(rows)  # Write the rows for this class

    print("Separate CSV files created for each unique class with only band-related columns.")