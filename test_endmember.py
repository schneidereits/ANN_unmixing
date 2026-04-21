#!/usr/bin/env python3
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Load prm
from prm import prm_demo_PLF_STM as prm
sys.modules['prm'] = prm

# Import and run the script
from scripts import script_01_endmember_filter_and_format
script_01_endmember_filter_and_format.main()

print("\n✓ Endmember export completed successfully!")
