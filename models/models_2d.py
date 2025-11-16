import torch
cuda_device = torch.device("cuda")
from lib.modules.multiblock_cnn import MultiblockConv,MultiblockReLU,MultiblockConv_naive,MultiblockLeakyReLU
import PISOtorch # core CUDA module
from solvers.solver_2d import PISOtorch_diff # differentiable wrapper for PISO
from trainer.Loss import ttonp
from lib.data.BlocksHandling import combine_list

class SmallCNNModel(torch.nn.Module):
    def __init__(self, input_channel, domain, dtype):
        super().__init__()
        self.dims = domain.getSpatialDims()
        self.leaky_relu = MultiblockLeakyReLU()
        self.conv1 = MultiblockConv(domain, "ZERO", input_channel, 16, kernel_size=7, padding="same", device=cuda_device, dtype=dtype)
        self.conv2 = MultiblockConv(domain, "ZERO", 16, 32, kernel_size=5, padding="same", device=cuda_device, dtype=dtype)
        self.conv3 = MultiblockConv(domain, "ZERO", 32, 64, kernel_size=5, padding="same", device=cuda_device, dtype=dtype)
        self.conv4 = MultiblockConv(domain, "ZERO", 64, 64, kernel_size=3, padding="same", device=cuda_device, dtype=dtype)
        self.conv5 = MultiblockConv(domain, "ZERO", 64, 64, kernel_size=3, padding="same", device=cuda_device, dtype=dtype)
        self.conv6 = MultiblockConv_naive(self.dims, 64, 64, kernel_size=1, padding="same", device=cuda_device, dtype=dtype)
        self.conv7 = MultiblockConv_naive(self.dims, 64, 2, kernel_size=1, padding="same", device=cuda_device, dtype=dtype)
        # self.convRes = MultiblockConv_naive(self.dims, input_channel, 64, kernel_size=1, padding="same", device=cuda_device, dtype=dtype)

    def forward(self, X, domain=None):
        Y = self.leaky_relu(self.conv1(X, domain=domain))
        Y = self.leaky_relu(self.conv2(Y, domain=domain))
        Y = self.leaky_relu(self.conv3(Y, domain=domain))
        Y = self.leaky_relu(self.conv4(Y, domain=domain))
        Y = self.leaky_relu(self.conv5(Y, domain=domain))
        Y = self.leaky_relu(self.conv6(Y, domain=domain))

        # Y = self.relu([y + x for y, x in zip(Y, self.convRes(X))])
        return self.conv7(Y, domain=domain)

# base class for the corrector
class CorrectorBase(torch.nn.Module):
    def __init__(self, stats=None) -> None:
        super().__init__()
        self.Re = None
        self.geo_feature = None # herein, it stands for the geometry feature, for BFS, it is the step height, for Karman, it is the obstacle y_in, and it varies, so this is just a placeholder
        self.correction = []
        self.input = []
        self.stats = stats
    
    def init_common(self, train_params, sim_params, LOG, domain):
        """
        Helper method to initialize attributes common to all correctors.
        """
        self.train_params = train_params
        self.sim_params = sim_params
        self.skip_blocks = train_params.skip_blocks
        self.dtype = train_params.dtype
        # assert domain is not None, "Domain must be provided"
        self.model = train_params.model(domain=domain,input_channel=train_params.input_channel, dtype=train_params.dtype)
        self.dims = domain.getSpatialDims()
        self.LOG = LOG
        self.layout = sim_params.layout
        self.data_norm = train_params.data_norm

    def mask_block(self,data_list):
        # for mask the buffer block, don't mask in-place
        mask = torch.tensor([i in self.skip_blocks for i in range(len(data_list))])
        return [torch.zeros_like(x) if m else x for x, m in zip(data_list, mask)]

    def norm_velocity(self):
        # only used before input to NN to avoid affect dataset
        normed_data_list=[]
        stats=self.stats
        if stats is None:
            raise ValueError("Stats must be provided for data normalization.")
        if self.data_norm == "std":
            # std seems not so good
            for data in self.input_tensors:
                if self.dims==2:
                    normed_data=torch.cat([
                        (data[:,0:1,...])/(stats["u_std"]),
                        (data[:,1:2,...])/(stats["v_std"])],dim=1)
                normed_data_list.append(normed_data)
            self.LOG.info(f"Data normalized by STD")
        elif self.data_norm == "max_min":
            for data in self.input_tensors:
                if self.dims==2:
                    normed_data=torch.cat([
                        (data[:,0:1,...]-stats["u_min"])/(stats["u_max"]-stats["u_min"]),
                        (data[:,1:2,...]-stats["v_min"])/(stats["v_max"]-stats["v_min"])],dim=1)
                elif self.dims==3:
                    normed_data=torch.cat([
                        (data[:,0:1,...]-stats["u_min"])/(stats["u_max"]-stats["u_min"]),
                        (data[:,1:2,...]-stats["v_min"])/(stats["v_max"]-stats["v_min"]),
                        (data[:,2:3,...]-stats["w_min"])/(stats["w_max"]-stats["w_min"])],dim=1)
                normed_data_list.append(normed_data)
            self.LOG.info(f"Data normalized by max_min")
        else:
            raise ValueError("Invalid data normalization method.")
        return normed_data_list
    
    def back_norm_velocity(self,data_list):
        # only used after output from NN with velocity correction method
        back_normed_data_list=[]
        stats=self.stats
        if stats is None:
            raise ValueError("Stats must be provided for data back-normalization.")
        if self.data_norm == "std":
            for data in data_list:
                if self.dims==2:
                    back_normed_data=torch.cat([
                        (data[:,0:1,...])*(stats["u_std"]),
                        (data[:,1:2,...])*(stats["v_std"])],dim=1)
                back_normed_data_list.append(back_normed_data)
            self.LOG.info(f"Data back-normalized by STD")
        elif self.data_norm == "max_min":
            for data in data_list:
                if self.dims==2:
                    back_normed_data=torch.cat([
                        (data[:,0:1,...]*(stats["u_max"]-stats["u_min"])+stats["u_min"]),
                        (data[:,1:2,...]*(stats["v_max"]-stats["v_min"])+stats["v_min"])],dim=1)
                elif self.dims==3:
                    back_normed_data=torch.cat([
                        (data[:,0:1,...]*(stats["u_max"]-stats["u_min"])+stats["u_min"]),
                        (data[:,1:2,...]*(stats["v_max"]-stats["v_min"])+stats["v_min"]),
                        (data[:,2:3,...]*(stats["w_max"]-stats["w_min"])+stats["w_min"])],dim=1)
                back_normed_data_list.append(back_normed_data)
            self.LOG.info(f"Data back-normalized by max_min")
        else:
            raise ValueError("Invalid data normalization method.")
        return back_normed_data_list

    
    def norm_feature(self):
        shapes = [[t.size(dim) for dim in range(t.dim())] for t in self.input_tensors]
        for shape in shapes:
            shape[1] = 1
        Re_tensor = [torch.ones(shape, device=cuda_device, dtype=self.dtype)*(self.Re / self.stats["Re_norm"]) for shape in shapes]
        geo_tensor = [torch.ones(shape, device=cuda_device, dtype=self.dtype)*(self.geo_feature / self.stats["geo_norm"]) for shape in shapes]
        
        return Re_tensor, geo_tensor

    def init_correction(self):
        self.correction = []
        self.input = []
    
    def forward(self, X, domain=None):
        return self.model(X, domain=domain)


class CorrectorINC(CorrectorBase):
    def __init__(self,train_params,sim_params,LOG,domain=None) -> None:
        super().__init__()
        self.additive = False
        self.init_common(train_params, sim_params, LOG, domain) 
        self.back_data_norm = False
        # never back norm for force as the scale is not related to velocity

    def correct_prediction(self, domain, **kwargs):
        # data preparation
        orig_vels = [block.velocity for block in domain.getBlocks()]
        self.input_tensors = orig_vels if self.skip_blocks is None else self.mask_block(orig_vels)
        if self.data_norm is not None:
            self.input_tensors=self.norm_velocity()
        Re_tensor, geo_tensor = self.norm_feature()
        self.input_tensors = [torch.cat([t, r, s], dim=1) for t, r, s in zip(self.input_tensors, Re_tensor, geo_tensor)]
        self.input.append(ttonp(combine_list(self.input_tensors,self.layout)))
        
        # run network
        x=self.forward(self.input_tensors, domain=domain)
        self.correction.append(ttonp(combine_list(x,self.layout)))
        if self.back_data_norm:
            x = self.back_norm_velocity(x)
        x = self.mask_block(x) if self.skip_blocks is not None else x

        # set NN output as VelocitySource, via blocks
        for block, t in zip(domain.getBlocks(), x):
            block.setVelocitySource(t)
        domain.UpdateDomainData()

        # re-set pre-advection velocity to blocks so that no changes to the initial velocity
        for block, orig_vel in zip(domain.getBlocks(), orig_vels):
            block.setVelocity(orig_vel)
        domain.UpdateDomainData()