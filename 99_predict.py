# 99_prm_predict.py
# Modular training script that orchestrates multiple project steps by importing
# project parameters from prm.py and dynamically executing the relevant scripts.

import sys
import os
from pathlib import Path
import argparse
import importlib.util
import subprocess


###############################################################
# PRM
###############################################################
PRM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prm")
PRM_MODULE = PRM_DIR + "prm_fire_libs_CHAR_filtered"  # Default prm module name (can be overridden by env var or command-line arg)
# Set desired prm module name here (change to e.g. 'prm_CHAR', 'prm_reduced', or 'prm').
# This top-level value will be used unless overridden by the environment variable
# PRM_MODULE or the command-line flag --prm.

#---------------------------------------------------------------------

# Function to load prm module
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def load_prm(prm_name: str):
    """Dynamically load a prm module by name and expose it as 'prm'."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prm_file = os.path.join(script_dir, f"{prm_name}.py")
    
    if not os.path.exists(prm_file):
        raise FileNotFoundError(f"Could not find prm file: {prm_file}")

    spec = importlib.util.spec_from_file_location("prm", prm_file)
    module = importlib.util.module_from_spec(spec)

    # Predefine variables expected by the prm module
    module.BASE_DIR = os.path.abspath(os.path.dirname(prm_file))
    module.NAME = prm_name.split("prm_", 1)[1] if prm_name.startswith("prm_") else prm_name
    module.OUTPUT_ROOT = os.path.join(module.BASE_DIR, module.NAME)

    spec.loader.exec_module(module)

    # Expose as 'prm' for downstream imports
    sys.modules['prm'] = module
    return module

# -----------------------------
# Parse command-line or environment override
# -----------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--prm', default=os.environ.get('PRM_MODULE', PRM_MODULE))
args, _ = parser.parse_known_args()
prm_module_name = args.prm

# Load prm module in parent process
prm = load_prm(prm_module_name)

###############################################################
# MAIN FUNCTION
###############################################################

def main():
    """
    Sequentially run the project scripts:
      - Multiprocessing scripts are run as subprocesses with the correct PRM_MODULE
      - Other scripts are dynamically imported and executed if they have a main() function
    """
    # List of scripts to execute
    scripts = [
        "04_predict_parallel.py",
        "05_mosaic_frac.py",
        "05_mosaic_frac_time_series.py",
        "05_predict_stats.py",
        "05_visualization/time_series_graph/99_sampling_values_library_multiprocessing.py",
        "05_visualization/time_series_graph/99_plot_time_series_line.py",
        "05_visualization/time_series_animation/02_enmapcube_visualization_and_animation_fcover_image_cc.py",
    ]

    # Scripts that require subprocess execution
    subprocess_scripts = [
        "04_predict_parallel.py",
        "05_predict_stats.py",
        "05_visualization/time_series_graph/99_sampling_values_library_multiprocessing.py",
    ]

    for script in scripts:
        script_path = os.path.join(BASE_DIR, "scripts/", script)
        print("\n" + "=" * 50)
        print(f"Starting execution of {script}")
        print("=" * 50)

        if script in subprocess_scripts:
            try:
                print(f"Running {script} as a subprocess...")
                env = os.environ.copy()
                existing = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(BASE_DIR), existing]))
                env['PRM_MODULE'] = prm_module_name  # Pass dynamic prm module

                subprocess.run([sys.executable, script_path], check=True, env=env, cwd=BASE_DIR)

                print(f"Finished {script} subprocess execution")
            except Exception as e:
                import traceback
                print(f"Error running {script} as subprocess: {e}")
                traceback.print_exc()
            continue

        # Dynamically import non-subprocess scripts
        try:
            spec = importlib.util.spec_from_file_location(script[:-3], script_path)
            if spec is None or spec.loader is None:
                print(f"Cannot import {script}: spec or loader is None")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"Successfully imported {script}")

            if hasattr(module, "main"):
                print(f"Running {script}.main()")
                module.main()
                print(f"Finished {script}.main()")
            else:
                print(f"No main() function found in {script}")

        except Exception as e:
            import traceback
            print(f"Error running {script}: {e}")
            traceback.print_exc()

    print("\nAll scripts have been executed.")


if __name__ == "__main__":
    main()

