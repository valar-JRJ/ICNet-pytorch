"""Utility functions."""
from .loss import ICNetLoss
from .lr_scheduler import IterationPolyLR, ConstantLR
from .metric import runningScore, averageMeter
from .logger import SetupLogger
from .visualize import get_color_pallete