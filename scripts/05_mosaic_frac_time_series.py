################################################################################
#                                  Import                                      #
################################################################################

import os
import re
import shutil
from os import mkdir
from os.path import isdir
from osgeo import gdal
from prm import PREDICTIONS_DIR, PRODUCTS, CLASSES


################################################################################
#                                  User Settings                               #
################################################################################

cube_dir = PREDICTIONS_DIR
products = PRODUCTS

BAND_NAMES = CLASSES  # Must match band order in each TIF

################################################################################
#                           Function Definitions                               #
################################################################################

RED   = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def create_subfolder(directory, subfolder_name):
    print('\n--> Create subfolder')
    subfolder_path = os.path.join(directory, subfolder_name)
    if not isdir(subfolder_path):
        mkdir(subfolder_path)
        print(f'    ... Created subfolder: {subfolder_name}')
    else:
        print(f'{RED}    ... Subfolder already exists: {subfolder_name}{RESET}')
    return subfolder_path


def extract_date(filename):
    """Extract the date string (YYYYMMDD) from a filename like ENMAPL2A_20230313_VEGCOV_FRAC.vrt"""
    match = re.search(r'(\d{8})', filename)
    return match.group(1) if match else None


def get_stem(filename):
    """Return filename without extension."""
    return os.path.splitext(filename)[0]


def collect_mosaic_vrts(mosaic_dir, products):
    """
    Collect all per-date VRTs from the mosaic/ folder, grouped by product suffix.
    Returns a dict: { product_ext: [ (date_str, full_vrt_path), ... ] }
    sorted by date ascending.
    """
    print('\n--> Collecting per-date VRTs from mosaic folder')
    result = {p: [] for p in products}

    for f in sorted(os.listdir(mosaic_dir)):
        for product in products:
            product_vrt = os.path.splitext(product)[0] + '.vrt'
            if f.endswith(product_vrt):
                date = extract_date(f)
                if date:
                    result[product].append((date, os.path.join(mosaic_dir, f)))

    for product in products:
        result[product].sort(key=lambda x: x[0])
        print(f'    ... {product}: found {len(result[product])} VRTs')

    return result


def get_raster_info(vrt_path):
    """Open a VRT and return (xsize, ysize, geotransform, projection, nodata, datatype_str)."""
    ds = gdal.Open(vrt_path)
    if ds is None:
        raise RuntimeError(f'Could not open: {vrt_path}')
    xsize      = ds.RasterXSize
    ysize      = ds.RasterYSize
    geotrans   = ds.GetGeoTransform()
    projection = ds.GetProjection()
    band       = ds.GetRasterBand(1)
    nodata     = band.GetNoDataValue()
    dtype_str  = gdal.GetDataTypeName(band.DataType)
    ds = None
    return xsize, ysize, geotrans, projection, nodata, dtype_str


def build_timeseries_vrt_xml(out_vrt_path, date_vrt_pairs, band_idx, band_name, nodata, dtype_str):
    """
    Write a VRT file by hand so that every <SourceFilename> uses a path
    relative to the output VRT — no intermediate temp files required.

    Each entry in date_vrt_pairs is (date_str, mosaic_vrt_path).
    band_idx (1-based) selects which band to pull from each source VRT.
    """
    # Use the first source to get spatial reference
    first_vrt = date_vrt_pairs[0][1]
    xsize, ysize, gt, proj, _, _ = get_raster_info(first_vrt)

    out_vrt_dir = os.path.dirname(os.path.abspath(out_vrt_path))

    n_bands = len(date_vrt_pairs)
    nd_str  = str(int(nodata)) if nodata is not None and nodata == int(nodata) else str(nodata)

    lines = []
    lines.append(f'<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">')
    lines.append(f'  <SRS>{proj}</SRS>')
    lines.append(
        f'  <GeoTransform>{", ".join(str(v) for v in gt)}</GeoTransform>'
    )

    for out_band_idx, (date, src_vrt_path) in enumerate(date_vrt_pairs, start=1):
        rel_path = os.path.relpath(os.path.abspath(src_vrt_path), out_vrt_dir)
        lines.append(f'  <VRTRasterBand dataType="{dtype_str}" band="{out_band_idx}">')
        lines.append(f'    <Description>{date}</Description>')
        if nodata is not None:
            lines.append(f'    <NoDataValue>{nd_str}</NoDataValue>')
        lines.append(f'    <SimpleSource>')
        lines.append(f'      <SourceFilename relativeToVRT="1">{rel_path}</SourceFilename>')
        lines.append(f'      <SourceBand>{band_idx}</SourceBand>')
        lines.append(f'    </SimpleSource>')
        lines.append(f'  </VRTRasterBand>')

    lines.append('</VRTDataset>')

    with open(out_vrt_path, 'w') as f:
        f.write('\n'.join(lines))


def build_timeseries_vrts(vrts_by_product, mosaic_ts_dir, band_names):
    """
    For each product, write one VRT per band (GV, NPV, SOIL, SHADE).
    Each VRT has N bands — one per date (chronological), referencing the
    per-date mosaic VRTs in mosaic/ directly via hand-written XML.

    Output naming: ENMAPL2A_TSS_<BAND_NAME>_FRAC.vrt
    """
    print('\n--> Building time series VRTs')

    for product, date_path_list in vrts_by_product.items():
        if not date_path_list:
            print(f'{RED}    ... No VRTs found for product {product}, skipping{RESET}')
            continue

        first_fname = os.path.basename(date_path_list[0][1])
        parts       = get_stem(first_fname).split('_')
        prefix      = parts[0]   # e.g. ENMAPL2A
        suffix      = parts[-1]  # e.g. FRAC

        print(f'\n    Product: {product}  |  {len(date_path_list)} dates  |  {len(band_names)} bands')

        # Read nodata and dtype once from the first source VRT
        _, _, _, _, nodata, dtype_str = get_raster_info(date_path_list[0][1])

        for band_idx, band_name in enumerate(band_names, start=1):
            out_vrt_name = f'{prefix}_TSS_{band_name}_{suffix}.vrt'
            out_vrt_path = os.path.join(mosaic_ts_dir, out_vrt_name)

            if os.path.isfile(out_vrt_path):
                print(f'{RED}    ... VRT already exists: {out_vrt_name}{RESET}')
                continue

            print(f'    ... Writing: {out_vrt_name}')
            build_timeseries_vrt_xml(
                out_vrt_path   = out_vrt_path,
                date_vrt_pairs = date_path_list,   # [(date, mosaic_vrt_path), ...]
                band_idx       = band_idx,
                band_name      = band_name,
                nodata         = nodata,
                dtype_str      = dtype_str,
            )

        print(f'{GREEN}    ... Finished product: {product}{RESET}')


################################################################################
#                               Execution                                      #
################################################################################

def main():
    """
    Build per-band time series VRTs by writing VRT XML that references the
    existing per-date mosaic VRTs directly — no intermediate temp files.

    Inputs:  mosaic/<PREFIX>_<YYYYMMDD>_<PRODUCT>_<SUFFIX>.vrt  (4 bands each)
    Outputs: mosaic_timeseries/<PREFIX>_TSS_<BAND>_<SUFFIX>.vrt
               Band 1 = earliest date ... Band N = latest date
               Band description = YYYYMMDD date string
               SourceFilename points directly to mosaic/ VRT
    """
    print('\n=== Time Series VRT Builder Started ===')

    # Step 1: Resolve the existing mosaic/ subfolder (must already exist)
    mosaic_dir = os.path.join(cube_dir, 'mosaic')
    if not isdir(mosaic_dir):
        print(f'{RED}ERROR: mosaic/ folder not found at {mosaic_dir}{RESET}')
        print('Run the original mosaic script first to generate per-date VRTs.')
        return

    # Step 2: Create output subfolder
    mosaic_ts_dir = create_subfolder(cube_dir, 'mosaic_timeseries')

    # Step 3: Collect per-date VRTs from mosaic/, grouped by product
    vrts_by_product = collect_mosaic_vrts(mosaic_dir, products)

    # Step 4: Write one time series VRT per band with hand-crafted XML
    build_timeseries_vrts(vrts_by_product, mosaic_ts_dir, BAND_NAMES)

    print('\n=== Time Series VRT Builder Completed ===')


if __name__ == "__main__":
    main()
