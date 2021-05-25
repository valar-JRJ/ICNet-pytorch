"""Utility functions."""
from .loss import ICNetLoss
from .lr_scheduler import IterationPolyLR, ConstantLR, PloyStepLR
from .metric import runningScore, averageMeter
from .logger import SetupLogger, get_logger
from .visualize import get_color_pallete