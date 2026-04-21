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
PRM_DIR = "prm"
PRM_MODULE = "prm_demo_PLF_STM"
#PRM_MODULE = "prm_demo_veg_condition_time_series"
# Default prm module name (can be overridden by env var or command-line arg)
# Set desired prm module name here (change to e.g. 'prm_CHAR', 'prm_demo_PLF_STM', etc).
# This top-level value will be used unless overridden by the environment variable
# PRM_MODULE or the command-line flag --prm.

#---------------------------------------------------------------------

# Function to load prm module
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def load_prm(prm_name: str, output_base_location: str = None):
    """Dynamically load a prm module by name and expose it as 'prm'."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prm_file = os.path.join(script_dir, PRM_DIR, f"{prm_name}.py")
    
    if not os.path.exists(prm_file):
        raise FileNotFoundError(f"Could not find prm file: {prm_file}")

    spec = importlib.util.spec_from_file_location("prm", prm_file)
    module = importlib.util.module_from_spec(spec)
    
    # Explicitly set __file__ before executing module
    module.__file__ = prm_file
    module.__loader__ = spec.loader

    # Predefine variables expected by the prm module
    module.BASE_DIR = os.path.abspath(os.path.dirname(prm_file))
    # Set NAME to suffix after 'prm_' or default to 'prm'
    if prm_name.startswith("prm_"):
        module.NAME = prm_name.split("prm_", 1)[1]
    elif prm_name == "prm":
        module.NAME = "prm"
    else:
        module.NAME = prm_name
    
    # Handle OUTPUT_BASE_LOCATION override
    if output_base_location is not None:
        module.OUTPUT_BASE_LOCATION = output_base_location
    
    # Execute module (this will set OUTPUT_ROOT based on OUTPUT_BASE_LOCATION)
    spec.loader.exec_module(module)

    # Expose as 'prm' for downstream imports
    sys.modules['prm'] = module
    return module

# -----------------------------
# Parse command-line or environment override
# -----------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--prm', default=os.environ.get('PRM_MODULE', PRM_MODULE),
                   help='PRM module name to use')
parser.add_argument('--output-dir', default=os.environ.get('OUTPUT_BASE_LOCATION', None),
                   help='Base output directory for all outputs (optional)')
args, _ = parser.parse_known_args()
prm_module_name = args.prm
output_base_location = args.output_dir

# Load prm module in parent process with optional output directory override
prm = load_prm(prm_module_name, output_base_location)

# Print output configuration for user awareness
print(f"\n{'='*60}")
print(f"Configuration: {prm_module_name}")
print(f"Output Root: {prm.OUTPUT_ROOT}")
print(f"{'='*60}\n")

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
        "05_predict_stats.py"
    ]

    # Scripts that require subprocess execution
    subprocess_scripts = [
        "04_predict_parallel.py",
        "05_predict_stats.py",
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

