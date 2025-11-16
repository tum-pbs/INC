import torch
cuda_device = torch.device("cuda")
import torch.nn.functional as F
from lib.data.BlocksHandling import Get_velocity
import time
ttonp = lambda tensor: tensor.detach().cpu().numpy()

def Loss_PBDL(reference_data,current_data,u_std=1,v_std=1):
    u_error=F.mse_loss(reference_data[0,0,:,:],current_data[0,0,:,:],reduction='sum')/u_std**2
    v_error=F.mse_loss(reference_data[0,1,:,:],current_data[0,1,:,:],reduction='sum')/v_std**2
    return u_error+v_error

class LossAccumulator:
    def __init__(self, reference_domain,dtype,low_res_domain=None,layout=None, skip_blocks=None, LOG=None, stats=None):

        self.reference_domain = reference_domain
        self.low_res_domain = low_res_domain
        self.results = []
        self.time_stamps = None
        self.time_start = None
        self.layout=layout
        self.dtype = dtype
        self.skip_blocks=skip_blocks # [3,4] for BFS
        self.reset_loss()
        self.LOG = LOG
        self.stats = stats # for scaling the loss


    def reset_loss(self):
        self.loss = torch.zeros(1, dtype=self.dtype, device=cuda_device)
        self.steps = 0 # this steps is for the number of loss accumulated
        #LOG.info("loss reset")
    
    def add_loss(self, domain, out_it, **kwargs):
        # as log_fn to accumulate loss from output frames/steps
        if self.time_start is None:
            self.time_start = time.time()
        self.time_stamps = out_it
        loss_acc = 0
        num_cells = 0
        ref_domain=self.get_reference_domain(self.steps)
        for block_idx in range(domain.getNumBlocks()):
            #skip the buffer blocks
            if self.skip_blocks is not None and block_idx in self.skip_blocks:
                continue
            else:
                block = domain.getBlock(block_idx)
                ref_block=ref_domain.getBlock(block_idx)
                u_std=self.stats["u_std"] if self.stats is not None else 1
                v_std=self.stats["v_std"] if self.stats is not None else 1
                loss = Loss_PBDL(block.velocity, ref_block.velocity, u_std=u_std, v_std=v_std)
                if loss.isnan().any():
                    raise RuntimeError("loss is NaN.")
                loss_acc += loss
        self.record_results(ref_domain,domain,loss_acc)
        self.loss += loss_acc
        self.steps += 1

    
    def get_reference_domain(self, steps):
        return self.reference_domain[steps]

    def get_loss_normalized(self):
        if self.steps==0: raise RuntimeError("no loss recorded")
        return self.loss / self.steps
    
    def record_time_elapsed(self):
        time_end = time.time()
        time_elapsed = time_end - self.time_start
        self.time_start = time_end
        return time_elapsed

    def record_results(self,ref_domain,current_domain,loss):
        ref_u,ref_p=Get_velocity(ref_domain,self.layout)
        res_u,res_p=Get_velocity(current_domain,self.layout)
        self.results.append({
                "time_stamp": self.time_stamps,
                "ref_u": ttonp(ref_u),
                "res_u": ttonp(res_u),
                "ref_p": ttonp(ref_p),
                "res_p": ttonp(res_p),
                "loss": ttonp(loss),
                "time_elapsed": self.record_time_elapsed()
            })