"""
Configuration for 1D problems (Burgers, KS)
Split from config.py
"""
import torch
import numpy as np
from data.data_loader_1d import BurgersDataset, KSDataset, BG_collate_fn, KS_collate_fn

def init_random(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class KSSimParams():
    def __init__(self, domain_size, resolution=64, dt=1e-2,random_ratio=1,batch_size=None, device='cuda',time_scheme = 'ETRK2',total_time=50):
        self.dtype = torch.float64 # only used for simulation
        self.resolution = resolution
        self.dt = dt
        self.random_ratio = random_ratio # number of random initial conditions, with same domain size
        self.device = device
        self.time_scheme = time_scheme # chosen from ['ETRK2','etd1_step']
        self.total_time = total_time
        self.warm_up = 2000
        self.set_domain_size(domain_size)
        self.batch_size = len(self.domain_size) if batch_size is None else batch_size 
        self.init_alpha()
    
    def set_domain_size(self,domain_size):
        if isinstance(domain_size, (list, tuple)):
            domain_size = torch.tensor(domain_size, device=self.device)
        elif isinstance(domain_size, torch.Tensor):
            domain_size = domain_size.clone().detach()
        else:
            raise ValueError("domain_size must be a list, tuple, or tensor, but got:", type(domain_size))

        self.domain_size =  domain_size.repeat_interleave(self.random_ratio) #repeated for getting different initial conditions, then B = len(domain_size) * random_ratio
        
    def init_alpha(self):
        self.alpha =(torch.rand(self.batch_size,device = self.device)-0.5)

class KSTrainParams:
    def __init__(self,mstep=4, dt=1e-2, lr=1e-4, weight_decay=1e-7,lambda_gp=0,lambda_spectrum=0,epochs=100, test_steps=1000,batch_size=64,test_batch_size = 1500,starts_gap=10,model_type=None,correction_term=None, T_in=1,init_sim_fn=None):
        self.task = "KS"
        self.doublestep = True # KSsolver always use double step forward
        self.dataset = KSDataset
        self.collate_fn = KS_collate_fn
        self.init_sim_fn = init_sim_fn
        self.loss_fn = torch.nn.MSELoss()
        self.dtype = torch.float32
        self.starts_gap =starts_gap # for generating multiple trajectories with fixed gap, None for using dt/gen_dt
        self.seed =42
        self.dt = dt 
        self.device = 'cuda'
        self.model_type = model_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_gp=lambda_gp
        self.lambda_spectrum=lambda_spectrum
        self.epochs=epochs
        self.test_steps = test_steps//2 if self.doublestep else test_steps
        self.test_batch_size  = test_batch_size
        self.mstep = mstep # number of steps forward in training
        self.valid_step = 50 # number of steps for validation
        self.batch_size = batch_size
        self.train_ratio = 0.9
        self.valid_ratio = 0.1
        self.down_resolution = 64
        self.metadata = None  # metadata for simualtion paramters
        self.correction_term = correction_term
        self.lr_decay = 0.5
        self.lr_decay_patience = 4
        self.early_stop_patience = 15
        self.adaptive_CFL = None 
        self.modes = 16 
        self.model = None
        self.T_in = T_in
        init_random(self.seed)

    def init_model(self):
        from models.models_1d import (FNO1D, UNet1D, TinyCNNNet, DeepONet1D, ResNet1D, UNetMod1D,
                                     FNO1D_RNN, UNet1D_RNN, DeepONet1D_RNN, DeepONet1D_GRU, 
                                     ResNet1D_RNN, UNetMod1D_RNN)
        model_map = {"FNO": FNO1D,"UNet": UNet1D,"TinyCNN": TinyCNNNet,"DeepONet":DeepONet1D,"ResNet":ResNet1D, "UNetMod":UNetMod1D,"FNO1d_RNN": FNO1D_RNN,"UNet1D_RNN": UNet1D_RNN,"DeepONet1D_RNN":DeepONet1D_RNN,"DeepONet1D_GRU":DeepONet1D_GRU,"ResNet1D_RNN":ResNet1D_RNN,"UNetMod1D_RNN":UNetMod1D_RNN}
        if "FNO" in self.model_type:
            self.model = model_map[self.model_type](L=None, modes=self.modes).to(self.device)
        elif self.model_type in ["DeepONet1D_RNN","DeepONet1D_GRU"]:
            self.model = model_map[self.model_type](L=None, sensor_dim=self.down_resolution, T_in= self.T_in).to(self.device)
        elif self.model_type =="DeepONet":
            self.model = model_map[self.model_type](L=None, sensor_dim=self.down_resolution).to(self.device)
        elif self.model_type not in model_map:
            raise ValueError(f"Model type {self.model_type} not found in model_map")
        else:
            self.model = model_map[self.model_type](L=None).to(self.device)