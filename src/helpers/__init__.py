"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import sys
from pathlib import Path

# Add parent directory to path for relative imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Bring all modules to the same level as helpers
from . import stack_array
from . import shape_npz
from . import xyz2arr
from . import xyz2arr_multi
