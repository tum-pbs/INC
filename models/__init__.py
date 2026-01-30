"""
Neural network models for 1D and 2D problems
"""
# from .models_1d import *

__all__ = []

# Try to import 2D models (requires PISOtorch)
try:
    from .models_2d import SmallCNNModel, CorrectorINC
    __all__.extend(['SmallCNNModel', 'CorrectorINC'])
except ImportError:
    # PISOtorch not available, skip 2D models
    pass