"""
Configuration parameters for 1D and 2D problems
"""
# Import 1D configurations
try:
    from .config_1d import BGSimParams, BGTrainParams, KSSimParams, KSTrainParams
except ImportError:
    pass

# Import 2D configurations  
try:
    from .config_2d import BFSSimParams, BFSTrainParams, init_2D_model, init_corrector, init_dataset
except ImportError:
    pass