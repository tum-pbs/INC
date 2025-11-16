"""
Configuration for 2D problems (BFS, etc.)
Split from config.py
"""
import torch
from models.models_2d import CorrectorINC, SmallCNNModel

def init_random(seed):
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_2D_model(model_type):
    model_dict = {
        "SmallCNN": SmallCNNModel,
    }
    if model_type not in model_dict:
        raise ValueError(f"Invalid model_type '{model_type}'. Valid options are: {list(model_dict.keys())}")
    return model_dict[model_type]

def init_corrector(method):
    method_config = {
            "INC": {"corrector": CorrectorINC, "prep_fn_key":"PRE"},
        }
    if method not in method_config:
        raise ValueError(f"Invalid method '{method}'. Valid options are: {list(method_config.keys())}")
    return method_config[method]["corrector"], method_config[method]["prep_fn_key"]

### ----- for backward facing step ----- ###
class BFSSimParams:
    def __init__(self, Re,s, CFL=0.8, downsample_factor=1, h=1,substeps="ADAPTIVE"):
        self.downsample_factor = downsample_factor
        self.base_list = [[20, 2]] # For refinement of grid generation [x,y]
        self.geo_list = [s * h] 
        self.l = 5 * h
        self.L = 32 * h
        self.h = h
        self.H = h + s * h
        self.res_x = int( (16 * (32 * h)) / downsample_factor )
        base_res_y = int(self.H * 64)
        self.res_y = int(base_res_y / downsample_factor)
        self.res_l = int(64 / downsample_factor)
        self.in_vel = 1
        self.in_var = 0.4 # inflow velocity variation
        self.dims = 2
        self.dtype = torch.float32
        self.buffer_list = [3, int(32 / downsample_factor)] # for buffer blocks
        self.Re = Re
        self.func = "exp"
        self.vis_ratio = 10 # viscosity ratio used in buffer block
        self.time_step = 0.1 # if substeps is not ADAPTIVE, this is the fixed time step, otherwise, this is for recording the data save interval
        self.pressure_tol = 1e-6 # pressure solver tolerance
        self.substeps = substeps # number of substeps or "ADAPTIVE"
        self.adaptive_CFL=CFL # only valid for substeps="ADAPTIVE", else ignored
        self.layout = None


class BFSTrainParams:
    def __init__(self, mstep=4, lr=1e-4, weight_decay=0.5,epochs=10, test_steps=400,data_norm="max_min",method=None,model_type=None,scale_loss = False):
        self.dtype = torch.float32
        self.mstep = mstep # unrolled steps for training
        self.model_type =model_type 
        self.method=method
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs=epochs
        self.early_stop_patience = 10 # early stopping patience
        self.input_channel = 4 
        self.seed = 7
        self.start_steps = 3000 # for training and testing, skip the initial transient period
        self.valid_steps = 100 # validation steps
        self.test_steps = test_steps # total test steps
        self.train_range=(self.start_steps, None) 
        self.valid_range=(3000, None) 
        self.data_norm = data_norm
        self.skip_blocks = [3,4] # skip the buffer blocks
        self.scale_loss = scale_loss
        self.back_data_norm = False
        init_random(self.seed)
        self.model = init_2D_model(self.model_type) if self.model_type is not None else None
        self.Corrector, self.prep_fn_key = init_corrector(self.method)
        
        # Set default BFS parameters
        self.Re = 1400
        self.geo_feature = 1