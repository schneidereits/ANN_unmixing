import os
import sys
from prm import *
from prm import SPECTRAL_LIB, BAD_WAVELENGTHS_CSV, ENDMEMBER_DIR, CLASS_COL, FILTER_ENDMEMBERS, CLASSES, STM, STM_METRICS
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
    
    # Extract numeric values: for regular columns use as-is, for STM format extract wavelength
    numeric_cols_float = []
    numeric_cols_mapping = {}  # Map float value to actual column name for later retrieval
    
    for c in numeric_cols:
        try:
            # Try direct conversion (for regular numeric columns)
            val = float(c)
            numeric_cols_float.append(val)
            numeric_cols_mapping[val] = c
        except ValueError:
            # Try STM format: "wl_XXX.XX_YYY" - extract the wavelength part
            if isinstance(c, str) and c.startswith('wl_'):
                try:
                    wl_part = c.split('_')[1]  # Get "XXX.XX"
                    val = float(wl_part)
                    numeric_cols_float.append(val)
                    numeric_cols_mapping[val] = c
                except (IndexError, ValueError):
                    pass

    if not numeric_cols_float:
        raise ValueError("No numeric wavelength columns found in dataframe.")

    def find_closest_band(target, available_bands):
        closest_val = min(available_bands, key=lambda x: abs(x - target))
        return numeric_cols_mapping[closest_val]  # Return actual column name

    BAND_MAP = {str(b): find_closest_band(b, numeric_cols_float) for b in required_bands}

    print("Using closest available bands:")
    for k, v in BAND_MAP.items():
        print(f"  Target {k} nm -> Closest {v} nm")

    return BAND_MAP, numeric_cols

# ------------------------------------------------------------
# Endmember export function
# ------------------------------------------------------------
def save_endmembers(reflectance_resampled: pd.DataFrame, out_dir: str, wavelength_cols: list,
                    class_col: str = "class", bad_wavelengths_csv: str = None, stm_mode: bool = False):
    """Save class-wise endmembers as CSV files, excluding bad wavelengths."""
    os.makedirs(out_dir, exist_ok=True)

    # For STM mode, wavelength_cols already have the format "wl_XXX_percentile", so use as-is
    if stm_mode:
        print("STM mode: Using all wavelength+percentile columns as provided")
        good_wavelength_str = wavelength_cols
    else:
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


def plot_spectra_by_class(df: pd.DataFrame, wavelength_cols: list, class_col: str, out_dir: str, stm_mode: bool = False, stm_metrics: list = None):
    """
    Plot spectra by class. For STM mode, creates separate plots for each metric (p10, p25, p50, etc).
    For regular mode, creates individual and combined plots.
    
    Parameters:
    - df: DataFrame containing spectra and class labels.
    - wavelength_cols: list of columns representing wavelength bands.
    - class_col: column name for class labels.
    - out_dir: directory to save plots.
    - stm_mode: whether to use STM-specific plotting (one plot per metric).
    - stm_metrics: list of STM metrics (e.g., ['p10', 'p25', 'p50', 'p75', 'p90']).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert wavelength columns to numeric safely, handling STM format (wl_XXX.XX_YYY)
    wavelengths = []
    metric_dict = {}  # Map column name to metric (e.g., 'wl_370_p10' -> 'p10')
    
    for w in wavelength_cols:
        try:
            # Regular numeric column (e.g. "418.42")
            wavelengths.append(float(w))
        except (ValueError, TypeError):
            if isinstance(w, str):
                # STM format: "418.42_p10"  →  rsplit on last '_'
                parts = w.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        wavelengths.append(float(parts[0]))
                        metric_dict[w] = parts[1]   # e.g. 'p10'
                    except ValueError:
                        raise ValueError(f"Could not extract wavelength from column: {w}")
                else:
                    raise ValueError(f"Unexpected column format: {w}")
            else:
                raise ValueError(f"Could not parse column: {w}")        
    
    wavelengths = np.array(wavelengths)

    # Define colors for classes
    specific_colors = {
        "GV": "green", "NPV": "red", "SUB": "blue", "CHAR": "black",
        "btr": "#8B4513", "ntr": "#228B22", "shr": "#DAA520", "her": "#FF6347", "nve": "#4169E1"
    }
    class_colors = specific_colors.copy()
    
    classes = df[class_col].unique()
    palette_map = class_colors.copy()
    missing_classes = [c for c in classes if c not in palette_map]
    if missing_classes:
        default_palette = sns.color_palette("tab10", n_colors=max(len(missing_classes), 1))
        for i, c in enumerate(missing_classes):
            palette_map[c] = default_palette[i % len(default_palette)]

    
    # ========== STM MODE PLOTTING ========== 
    if stm_mode and stm_metrics:
        print("STM mode: Creating separate plots for each metric...")
        
        # Get ordered classes
        if 'CLASSES' in globals() and len(CLASSES) > 0:
            actual_classes = set(classes)
            ordered_classes = [c for c in CLASSES if c in actual_classes]
            ordered_classes += [c for c in classes if c not in CLASSES]
        else:
            ordered_classes = sorted(list(classes))
        
        # For each metric, create a faceted plot showing all classes
        # AFTER
        for metric in stm_metrics:
            # All columns whose suffix matches this metric  (e.g. "418.42_p10" → metric 'p10')
            metric_cols = [c for c in wavelength_cols if metric_dict.get(c) == metric]
            if not metric_cols:
                print(f"No columns found for metric '{metric}', skipping.")
                continue
            # Extract wavelength from the first part of the column name (e.g. "418.42_p10" → 418.42)
            metric_wavelengths = np.array([float(c.rsplit('_', 1)[0]) for c in metric_cols])
            
            # Create faceted plot: one subplot per class
            n_classes = len(ordered_classes)
            n_cols = 2
            n_rows = math.ceil(n_classes / n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()
            
            for i, cls in enumerate(ordered_classes):
                ax = axes[i]
                subset = df[df[class_col] == cls]
                
                if len(subset) == 0:
                    continue
                
                color = palette_map.get(cls, "gray")
                
                # Plot individual spectra for this class and metric
                for _, row in subset.iterrows():
                    ax.plot(metric_wavelengths, row[metric_cols].to_numpy(), color=color, alpha=0.2)
                
                # Plot mean spectrum
                mean_spec = subset[metric_cols].mean(axis=0)
                ax.plot(metric_wavelengths, mean_spec.to_numpy(), color="black", linewidth=3, label=f"{cls} mean")
                
                ax.set_title(f"{cls} ({metric})", fontsize=16)
                ax.grid(True, alpha=0.3)
                if i % n_cols == 0:
                    ax.set_ylabel("Reflectance", fontsize=14)
                if i >= (n_rows - 1) * n_cols:
                    ax.set_xlabel("Wavelength (nm)", fontsize=14)
                ax.legend(fontsize=10)
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                try:
                    fig.delaxes(axes[j])
                except (KeyError, ValueError):
                    pass
            
            plt.suptitle(f"Spectra by Class - Metric: {metric}", fontsize=18, y=0.995)
            plt.tight_layout()
            
            # Save plot
            metric_plot_path = os.path.join(out_dir, f"facet_spectra_{metric}.png")
            plt.savefig(metric_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"Saved faceted plot for metric '{metric}' at: {metric_plot_path}")
    
    # ========== REGULAR MODE PLOTTING ========== 
    else:
        print("Regular mode: Creating class-wise plots...")
        
        # Plot each class individually
        for cls in classes:
            subset = df[df[class_col] == cls]
            plt.figure(figsize=(10, 6))

            color = palette_map.get(cls, "gray")

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
            plt.show()
            plt.close()
            print(f"Saved plot for class '{cls}' at: {plot_path}")

        # Combined faceted plot of all classes
        if 'CLASSES' in globals() and len(CLASSES) > 0:
            actual_classes_in_data = set(classes)
            valid_classes = [c for c in CLASSES if c in actual_classes_in_data]
            ordered_classes = valid_classes + [c for c in classes if c not in CLASSES]
        else:
            ordered_classes = list(classes)
        
        n_classes = len(ordered_classes)
        n_cols = 2
        n_rows = math.ceil(n_classes / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, cls in enumerate(ordered_classes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            subset = df[df[class_col] == cls]
            
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
            try:
                fig.delaxes(axes[j])
            except (KeyError, ValueError):
                pass

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
        
        # Skip wavelength CSV matching when STM==True (use all band columns as-is)
        if STM:
            print("STM mode enabled: skipping wavelength CSV filtering. Using all band columns.")
            # Load all wavelengths and bad wavelengths to create meaningful band names
            all_wl_path = r"D:\ANN_unmixing\auxiliary\all_wavelengths.csv"
            all_wavelengths_df = pd.read_csv(all_wl_path, header=None)
            all_wavelengths = all_wavelengths_df.iloc[:, 0].tolist()
            
            # Load bad wavelengths
            bad_wl_path = r"D:\ANN_unmixing\auxiliary\bad_wavelengths.csv"
            bad_wavelengths = []
            if os.path.exists(bad_wl_path):
                bad_wl_df = pd.read_csv(bad_wl_path, header=None)
                bad_wavelengths = bad_wl_df.iloc[:, 0].round(2).tolist()
            
            # Filter to good wavelengths
            all_wavelengths_rounded = [round(w, 2) for w in all_wavelengths]
            good_wavelengths = [w for w in all_wavelengths_rounded if w not in bad_wavelengths]
            
            print(f"STM: Using {len(good_wavelengths)} wavelengths × 5 metrics = {len(good_wavelengths) * 5} bands")
            
            metrics = STM_METRICS
            stm_col_names = []
            for metric in metrics:
                for wl in good_wavelengths:
                    stm_col_names.append(f"{wl:.2f}_{metric}")
            
            # Verify we have the right number of columns
            if len(stm_col_names) == len(band_cols):
                wavelength_rename_map = {band_cols[i]: stm_col_names[i] for i in range(len(band_cols))}
                df = df.rename(columns=wavelength_rename_map)
                wavelength_cols = stm_col_names
                print(f"Retained all {len(wavelength_cols)} band columns with wavelength+percentile naming for STM processing")
            else:
                print(f"WARNING: Mismatch - expected {len(band_cols)} columns but got {len(stm_col_names)}")
                # Fallback: use simple numeric names
                wavelength_rename_map = {col: str(i) for i, col in enumerate(band_cols)}
                df = df.rename(columns=wavelength_rename_map)
                wavelength_cols = list(wavelength_rename_map.values())
        else:
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
    if not STM:
        BAND_MAP, numeric_cols = calc_ndvi_cai(df, wavelength_cols=wavelength_cols)
    else:
        BAND_MAP, numeric_cols = {}, wavelength_cols

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
        stm_mode=STM,
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

    # Pass STM parameters if available
    stm_metrics = getattr(sys.modules['prm'], 'STM_METRICS', None) if STM else None
    plot_spectra_by_class(df,
                          wavelength_cols,
                          CLASS_COL,
                          ENDMEMBER_DIR,
                          stm_mode=STM,
                          stm_metrics=stm_metrics)

    print("Processing complete.")

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
