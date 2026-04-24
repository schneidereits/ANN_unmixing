# 99_prm_train.py
# Modular training script that orchestrates multiple project steps by importing
# project parameters from prm.py and dynamically executing the relevant scripts.

import sys
import os
import pandas as pd
import argparse
import importlib.util


###############################################################
# PRM
###############################################################
PRM_DIR = "prm"
#PRM_MODULE = "prm_demo_taylor_Mt"
#PRM_MODULE = "prm_demo_taylor_Mt_with_eco"
PRM_MODULE = "prm_demo_PLF_STM"  
#PRM_MODULE = "prm_demo_data_cube"

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
# Handle command-line or env override
# -----------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--prm', default=os.environ.get('PRM_MODULE', PRM_MODULE),
                   help='PRM module name to use')
parser.add_argument('--output-dir', default=os.environ.get('OUTPUT_BASE_LOCATION', None),
                   help='Base output directory for all outputs (optional)')
args, _ = parser.parse_known_args()
prm_module_name = args.prm
output_base_location = args.output_dir

# Load the prm module with optional output directory override
prm = load_prm(prm_module_name, output_base_location)

# Print output configuration for user awareness
print(f"\n{'='*60}")
print(f"Configuration: {prm_module_name}")
print(f"Output Root: {prm.OUTPUT_ROOT}")
print(f"{'='*60}\n")



def main():
    """
    Main function to sequentially run the project scripts:
      1. 01_endmember_filter_and_format.py
      2. 02_synthmix.py
      3. 03_model_train.py

    Each script is dynamically imported using importlib, and if a 'main()' function
    exists in the script, it is executed. Errors are caught and printed without stopping
    the execution of subsequent scripts.
    """
    import importlib.util

    # List of scripts to execute in order
    scripts = [
        "01_endmember_filter_and_format.py",
        "02_synthmix.py",
        "03_model_train.py",
    ]

    for script in scripts:
        script_path = os.path.join(BASE_DIR, "scripts/", script)
        print("\n" + "=" * 50)
        print(f"Starting execution of {script}")
        print("=" * 50)

        try:
            # Dynamically import the script
            spec = importlib.util.spec_from_file_location(script[:-3], script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"Successfully imported {script}")

            # Run main() if it exists
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


