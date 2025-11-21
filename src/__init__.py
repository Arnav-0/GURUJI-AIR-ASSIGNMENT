"""
mmWave Radar AI Project
Core utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "mmWave Radar AI Team"

# Import modules (use try-except for flexibility)
try:
    from .signal_processing import *
    from .data_generator import *
    from .visualization import *
except ImportError:
    # Allow standalone imports
    pass
