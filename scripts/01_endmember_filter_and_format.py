import os
from prm import *
from prm import SPECTRAL_LIB, BAD_WAVELENGTHS_CSV, ENDMEMBER_DIR, CLASS_COL, FILTER_ENDMEMBERS, CLASSES, STM
import pandas as pd
import inspect
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Band mapping function
# ------------------------------------------------------------
def calc_ndvi_cai(df: pd.DataFrame, required_bands=[670, 800, 2000, 2100, 2200], wavelength_cols=None):
    """
    Map required bands to closest available numeric columns.
    Returns BAND_MAP dict and list of numeric columns.
    """
    if wavelength_cols is None:
        numeric_cols = [c for c in df.columns if str(c).replace('.', '', 1).isdigit()]
    else:
        numeric_cols = wavelength_cols
    
    numeric_cols_float = [float(c) for c in numeric_cols]

    if not numeric_cols:
        raise ValueError("No numeric wavelength columns found in dataframe.")

    def find_closest_band(target, available_bands):
        return str(min(available_bands, key=lambda x: abs(x - target)))

    BAND_MAP = {str(b): find_closest_band(b, numeric_cols_float) for b in required_bands}

    print("Using closest available bands:")
    for k, v in BAND_MAP.items():
        print(f"  Target {k} nm → Closest {v} nm")

    return BAND_MAP, numeric_cols

# ------------------------------------------------------------
# Endmember export function
# ------------------------------------------------------------
def save_endmembers(reflectance_resampled: pd.DataFrame, out_dir: str, wavelength_cols: list,
                    class_col: str = "class", bad_wavelengths_csv: str = None):
    """Save class-wise endmembers as CSV files, excluding bad wavelengths."""
    os.makedirs(out_dir, exist_ok=True)

    # Load bad wavelengths
    bad_wavelengths = []
    if bad_wavelengths_csv and os.path.exists(bad_wavelengths_csv):
        bad_wavelengths = pd.read_csv(bad_wavelengths_csv, header=None).iloc[:, 0].round(2).tolist()
    elif bad_wavelengths_csv is None:
        print("INFO: BAD_WAVELENGTHS_CSV is None. All wavelengths will be retained.")
    elif not os.path.exists(bad_wavelengths_csv):
        print(f"WARNING: BAD_WAVELENGTHS_CSV path does not exist: {bad_wavelengths_csv}. All wavelengths will be retained.")

    # Filter good wavelengths
    wl_cols_float = [round(float(w), 2) for w in wavelength_cols]
    good_wavelengths = [wl for wl in wl_cols_float if wl not in bad_wavelengths]
    good_wavelength_str = [f"{wl:.2f}" for wl in good_wavelengths]
    
    # Verify good wavelengths and print
    if len(good_wavelengths) == len(wl_cols_float) - len(bad_wavelengths):
        print(f"wavelengths subset sucessfully: {len(good_wavelengths)} out of {len(wl_cols_float)} remaining")
    else:
        print("Bad wavelengths removal issue")

    # Check missing columns
    missing = [w for w in good_wavelength_str if w not in reflectance_resampled.columns]
    if missing:
        raise KeyError(f"Missing wavelength columns: {missing}")

    # Save each class
    for cls in reflectance_resampled[class_col].unique():
        subset = reflectance_resampled[reflectance_resampled[class_col] == cls]
        subset_out = subset[good_wavelength_str].copy() #* 10000  # scale reflectance

        # Sanitize filename
        safe_cls = re.sub(r'[\\/:"*?<>|]+', "_", str(cls))
        out_path = os.path.join(out_dir, f"{safe_cls}.csv")
        subset_out.to_csv(out_path, index=False, header=False)

        # Report count
        num_rows = len(subset_out)
        print(f"for class '{cls}' ({num_rows} spectra)  Saved endmember: {out_path}")


def plot_spectra_by_class(df: pd.DataFrame, wavelength_cols: list, class_col: str, out_dir: str):
    """
    Plot all spectra row-wise for each class, save individual class plots, 
    show them in console, and create a combined plot of all classes.
    
    Parameters:
    - df: DataFrame containing spectra and class labels.
    - wavelength_cols: list of columns representing wavelength bands.
    - class_col: column name for class labels.
    - out_dir: directory to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert wavelength columns to numeric safely
    try:
        wavelengths = np.array([float(w) for w in wavelength_cols])
    except ValueError:
        raise ValueError("Wavelength columns contain non-numeric values.")

    # Define specific colors for known classes
    specific_colors = {
        "GV": "green",
        "NPV": "red",
        "SUB": "blue",
        "CHAR": "black"
    }

    # Build class_colors dynamically from CLASSES, using specific colors where available
    class_colors = specific_colors.copy()
    missing_classes = [c for c in CLASSES if c not in class_colors]
    if missing_classes:
        # Assign colors from a default palette for additional classes
        default_palette = sns.color_palette("tab10", n_colors=len(missing_classes))
        for c, color in zip(missing_classes, default_palette):
            class_colors[c] = color

    # Build palette map: if any classes are missing from `class_colors`, assign
    # them colors from a default seaborn palette so all classes get a color.
    classes = df[class_col].unique()
    palette_map = class_colors.copy()
    missing_classes = [c for c in classes if c not in palette_map]
    if missing_classes:
        default_palette = sns.color_palette("tab10", n_colors=max(len(missing_classes), 1))
        for i, c in enumerate(missing_classes):
            palette_map[c] = default_palette[i % len(default_palette)]

    # -----------------------------
    # Plot each class individually
    # -----------------------------
    for cls in classes:
        subset = df[df[class_col] == cls]
        plt.figure(figsize=(10, 6))

        color = palette_map.get(cls, "gray")  # default gray if still missing

        # Plot each row (spectrum)
        for _, row in subset.iterrows():
            plt.plot(wavelengths, row[wavelength_cols].to_numpy(), color=color, alpha=0.5)

        # Plot bold mean spectrum
        mean_spec = subset[wavelength_cols].mean(axis=0)
        plt.plot(wavelengths, mean_spec.to_numpy(), color=color, linewidth=2.2, label=f"{cls} mean")

        plt.title(f"Spectra for Class: {cls}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.grid(True)
        plt.legend()

        # Sanitize filename
        safe_cls = re.sub(r'[\\/:"*?<>|]+', "_", str(cls))
        plot_path = os.path.join(out_dir, f"{safe_cls}_spectra_plot.png")
        plt.savefig(plot_path, dpi=300)

        plt.show()  # display in console

        plt.close()
        print(f"Saved plot for class '{cls}' at: {plot_path}")

    # -----------------------------
    # Combined plot of all classes
    # -----------------------------
    # Get actual unique classes from the data
    classes = df[class_col].unique()
    
    # If CLASSES is defined and has fewer classes, use it to maintain order
    # Otherwise, use the actual classes in the data
    if 'CLASSES' in globals() and len(CLASSES) > 0:
        # Ensure all classes in CLASSES exist in the data
        actual_classes_in_data = set(classes)
        valid_classes = [c for c in CLASSES if c in actual_classes_in_data]
        # Add any missing classes from data
        missing_classes = [c for c in classes if c not in CLASSES]
        ordered_classes = valid_classes + missing_classes
    else:
        ordered_classes = list(classes)
    
    n_classes = len(ordered_classes)
    n_cols = 2  # number of columns in the facet layout
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Define specific colors for known classes
    specific_colors = {
        "GV": "green",
        "NPV": "red",
        "SUB": "blue",
        "CHAR": "black"
    }

    # Build class_colors dynamically from ordered_classes, using specific colors where available
    class_colors = specific_colors.copy()
    missing_classes = [c for c in ordered_classes if c not in class_colors]
    if missing_classes:
        # Assign colors from a default palette for additional classes
        default_palette = sns.color_palette("tab10", n_colors=len(missing_classes))
        for c, color in zip(missing_classes, default_palette):
            class_colors[c] = color

    # Build palette map: if any classes are missing from `class_colors`, assign
    # them colors from a default seaborn palette so all classes get a color.
    palette_map = class_colors.copy()
    missing_classes = [c for c in ordered_classes if c not in palette_map]
    if missing_classes:
        default_palette = sns.color_palette("tab10", n_colors=max(len(missing_classes), 1))
        for i, c in enumerate(missing_classes):
            palette_map[c] = default_palette[i % len(default_palette)]

    for i, cls in enumerate(ordered_classes):
        if i >= len(axes):
            break
            
        ax = axes[i]
        subset = df[df[class_col] == cls]
        
        # Skip empty subsets
        if len(subset) == 0:
            continue
            
        color = palette_map.get(cls, "gray")

        # Plot individual spectra
        for _, row in subset.iterrows():
            ax.plot(wavelengths, row[wavelength_cols].to_numpy(), color=color, alpha=0.2)

        # Plot mean spectrum
        mean_spec = subset[wavelength_cols].mean(axis=0)
        ax.plot(wavelengths, mean_spec.to_numpy(), color="black", linewidth=3, label=f"{cls} mean")

        ax.set_title(cls, fontsize=18)
        ax.grid(True)
        if i % n_cols == 0:
            ax.set_ylabel("Reflectance", fontsize=16)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Wavelength (nm)", fontsize=16)
        ax.legend(fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Spectra by Class", fontsize=18, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    facet_path = os.path.join(out_dir, "facet_spectra_plot.png")
    plt.savefig(facet_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved faceted plot at: {facet_path}")

    
    ####################################
    # NDVI vs CAI plot
    #######################################
    if STM:
        print("NDVI or CAI columns not found. Skipping NDVI vs CAI plot.")
        return
    df = df[df[CLASS_COL].isin(CLASSES)].copy()
    # palette_map is reused here to ensure all classes have colors

    # Compute NDVI and CAI using BAND_MAP
    # Find closest numeric columns to required wavelengths
    numeric_cols = [c for c in df.columns if str(c).replace(".", "", 1).isdigit()]
    numeric_cols_float = [float(c) for c in numeric_cols]

    def find_closest_band(target):
        return str(min(numeric_cols_float, key=lambda x: abs(x - target)))

    red_band = find_closest_band(670)
    nir_band = find_closest_band(800)
    cai1 = find_closest_band(2000)
    cai2 = find_closest_band(2200)
    cai3 = find_closest_band(2100)

    # Compute NDVI and CAI
    df['NDVI'] = (df[nir_band] - df[red_band]) / (df[nir_band] + df[red_band])
    df['CAI'] = 0.5 * (df[cai1] + df[cai2]) - df[cai3]

    # --- Feature plot by class ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='NDVI', y='CAI',   hue=CLASS_COL, palette=palette_map, alpha=0.7, s=50)
    plt.title("Feature Space: NDVI vs CAI by Class", fontsize=22)
    plt.xlabel("NDVI", fontsize=14)
    plt.ylabel("CAI", fontsize=14)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    feature_class_path = os.path.join(ENDMEMBER_DIR, "feature_space_NDVI_CAI_class.png")
    plt.savefig(feature_class_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved NDVI vs CAI (by class) plot at: {feature_class_path}")




# ------------------------------------------------
# Main Execution
# ------------------------------------------------------------
def main():
    print("Loading resampled reflectance data...")
    df = pd.read_csv(SPECTRAL_LIB)

    # Identify wavelength columns
    # First check for columns containing "band"
    band_cols = [c for c in df.columns if 'band' in str(c).lower()]
    
    if band_cols:
        print(f"Found {len(band_cols)} band columns")
        
        # Load all wavelengths from CSV
        all_wl_path = r"D:\ANN_unmixing\auxiliary\all_wavelengths.csv"
        all_wavelengths_df = pd.read_csv(all_wl_path, header=None)
        all_wavelengths = all_wavelengths_df.iloc[:, 0].tolist()
        
        # Load bad wavelengths and filter
        bad_wl_path = r"D:\ANN_unmixing\auxiliary\bad_wavelengths.csv"
        bad_wavelengths = []
        if os.path.exists(bad_wl_path):
            bad_wl_df = pd.read_csv(bad_wl_path, header=None)
            bad_wavelengths = bad_wl_df.iloc[:, 0].round(2).tolist()
        
        # Match length: if lengths match, filter bad wavelengths; otherwise use all
        if len(bad_wavelengths) > 0 and len(bad_wavelengths) <= len(all_wavelengths):
            all_wavelengths_rounded = [round(w, 2) for w in all_wavelengths]
            good_wavelengths = [w for w in all_wavelengths_rounded if w not in bad_wavelengths]
            print(f"Filtered to {len(good_wavelengths)} wavelengths (excluded {len(bad_wavelengths)} bad bands)")
        else:
            good_wavelengths = [round(w, 2) for w in all_wavelengths]
            print(f"Using all {len(good_wavelengths)} wavelengths from all_wavelengths.csv")
        
        # Create mapping: band_col → wavelength, and wavelength_cols as band names
        # Rename columns to wavelength values for consistency with rest of pipeline
        wavelength_rename_map = {band_cols[i]: f"{good_wavelengths[i]:.2f}" for i in range(len(band_cols)) if i < len(good_wavelengths)}
        df = df.rename(columns=wavelength_rename_map)
        wavelength_cols = list(wavelength_rename_map.values())
        print(f"Renamed {len(wavelength_cols)} band columns to wavelength values")
    else:
        print("No band columns found. Looking for numeric columns...")
        wavelength_cols = [c for c in df.columns if str(c).replace(".", "", 1).isdigit()]
        print(f"Found {len(wavelength_cols)} numeric wavelength bands.")

    # Map bands
    BAND_MAP, numeric_cols = calc_ndvi_cai(df, wavelength_cols=wavelength_cols)

    # Apply filters if requested
    if FILTER_ENDMEMBERS:
        print("Applying custom filters...")
        df = filter_endmembers(df, BAND_MAP, numeric_cols)

        # Save filter function for reproducibility
        os.makedirs(ENDMEMBER_DIR, exist_ok=True)
        filter_code_path = os.path.join(ENDMEMBER_DIR, "applied_filters_function.txt")
        filter_code = inspect.getsource(filter_endmembers)
        with open(filter_code_path, 'w') as f:
            f.write(filter_code)

    # Export endmembers
    print("Exporting endmembers by class...")
    save_endmembers(
        reflectance_resampled=df,
        out_dir=ENDMEMBER_DIR,
        wavelength_cols=wavelength_cols,
        class_col=CLASS_COL,
        bad_wavelengths_csv=BAD_WAVELENGTHS_CSV,
    )
    
     # -----------------------------
    # Create and print table: counts per class per source
    # -----------------------------

# -----------------------------
# Create and print table: counts per class per source
# -----------------------------
    if 'source' in df.columns:
        # Group and count
        summary_table = df.groupby('source')[CLASS_COL].value_counts().unstack(fill_value=0)

        # Add total column per source
        summary_table['Total'] = summary_table.sum(axis=1)

        # Sort rows by total descending
        summary_table = summary_table.sort_values(by='Total', ascending=False)

        print("\n=== Counts per class per source ===\n")
        # Use pandas built-in display
        print(summary_table)

        # Save CSV as well
        table_path = os.path.join(ENDMEMBER_DIR, "counts_per_class_per_source.csv")
        summary_table.to_csv(table_path)
        print(f"\nSaved summary table at: {table_path}")
        
    # -----------------------------
# Create and print table: counts per class per category_1
# -----------------------------
    if 'category_1' in df.columns:
        # Group and count
        summary_table = df.groupby('category_1')[CLASS_COL].value_counts().unstack(fill_value=0)

        # Add total column per category
        summary_table['Total'] = summary_table.sum(axis=1)

        # Sort rows by total descending
        summary_table = summary_table.sort_values(by='Total', ascending=False)

        print("\n=== Counts per class per source ===\n")
        # Use pandas built-in display
        print(summary_table)

        # Save CSV as well
        table_path = os.path.join(ENDMEMBER_DIR, "counts_per_class_per_category_1.csv")
        summary_table.to_csv(table_path)
        print(f"\nSaved summary table at: {table_path}")

    plot_spectra_by_class(df,
                          wavelength_cols,
                          CLASS_COL,
                          ENDMEMBER_DIR)

    print("Processing complete.")

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
