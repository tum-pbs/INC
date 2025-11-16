import torch
import PISOtorch # domain data structures and core PISO functions. check /extensions/PISOtorch.cpp to see what is available
from solvers.solver_2d import PISOtorch_simulation # helper for PISO loop
import lib.data.shapes as shapes
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")
import numpy as np
# from BlocksHandling import get_block_shape_by_grid,combine_transforms

def reset_domain(domain):
    domain.CreateVelocityOnBlocks()
    domain.CreatePressureOnBlocks()
    # domain.CreatePassiveScalarOnBlocks()
    domain.PrepareSolve()
    
def clean_domain(domain,prep_fn):
    for block in domain.getBlocks():
        block.MakeAllPeriodic()
    del domain
    del prep_fn
  

class DomainData:
    def __init__(self, domain, time_stamp=None, grids=None, layout=None, prep_fn=None):
        self.domain = domain  
        self.time_stamp = time_stamp  # Time stamp for the domain sample
        self.grids=grids
        self.layout=layout
        self.prep_fn=prep_fn


class BFSDomainManager:
    def __init__(self, **kwargs):
        """
        Initializes the BFSDomainManager and loads configurations based on the provided parameters.
        This class is used for simulation and learning
        Args:
            **kwargs: Dictionary of parameters for the solver configuration.
        """
        # Dynamically set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Additional setup if necessary
        self.domain_dict = {}
        self._initialize_configs()

    def _initialize_configs(self):
        """
        Internal method to populate the domain_dict with DomainData instances. (new version with in/out flow inside the makedomain function)
        """
        for base in self.base_list:
            for s in self.geo_list:
                H = self.h + s
                self.viscosity = self.in_vel * 2 * self.h / self.Re
                # Generate domain configuration
                domain, layout, BFS_prep_fn, grids = makeBFSdomain_BufferBlock(
                    l=self.l, L=self.L, H=H, h=self.h, base=base, in_vel=self.in_vel, in_var=self.in_var, res_x=self.res_x, 
                    res_y=self.res_y, res_l=self.res_l, dims=self.dims, viscosity=self.viscosity, 
                    dtype=self.dtype, func=self.func, buffer_list=self.buffer_list, vis_ratio=self.vis_ratio
                )
                
                # Create DomainData instance and store it in the dictionary
                config = DomainData(domain=domain, grids=grids, prep_fn=BFS_prep_fn, layout=layout)
                self.domain_dict[(tuple(base), s, self.Re)] = config

    def get_config(self, key):
        return self.domain_dict.get(key, None)
    
    def get_all_keys(self):
        return list(self.domain_dict.keys())
    
    def __getitem__(self, key):
        if key in self.domain_dict:
            return self.domain_dict[key]
        raise KeyError(f"Configuration key {key} not found.")

    def __contains__(self, key):
        return key in self.domain_dict

class KarmanDomainManager:
    def __init__(self, **kwargs):
        """
        Initializes the DomainManager and loads configurations based on the provided parameters.
        similar to BFSDomainManager
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.domain_dict = {}
        self._initialize_configs()
    def _initialize_configs(self):
        for base in self.base_list: 
            for y_in in self.geo_list:
                y_in_res = int(y_in*self.res_scale)
                self.viscosity = y_in*self.in_vel/self.Re
                domain, prep_fn, layout,grids = make8BlockChannelFlowSetup(x=self.res_x, y=self.res_y, z=None, x_in=self.x_in, y_in=y_in_res, x_pos=self.x_pos, in_vel=self.in_vel, in_var=self.in_var, closed_y=True, closed_z=True, viscosity=self.viscosity, scale=1.0/self.res_scale, dtype=self.dtype, base=base, denser=self.denser, disturbance=True)
                config = DomainData(domain=domain, grids=grids, prep_fn=prep_fn, layout=layout)
                self.domain_dict[(tuple(base), y_in, self.Re)] = config
    
    def get_config(self, key):
        return self.domain_dict.get(key, None)
    
    def get_all_keys(self):
        return list(self.domain_dict.keys())
    
    def __getitem__(self, key):
        if key in self.domain_dict:
            return self.domain_dict[key]
        raise KeyError(f"Configuration key {key} not found.")

    def __contains__(self, key):
        return key in self.domain_dict

def make5BFSGrid(l,L,H,h,base,res_x,res_y,buffer_list,dtype,func,res_l=64):

    buffer_length=buffer_list[0]
    buffer_res=buffer_list[1]
    res_h=int(res_y*h/H)
    res_H=res_y
    res_L=res_x
    x_sizes=[res_l,res_L]
    y_sizes=[res_h,res_H-res_h]
    y_poss=[0,H-h,H]
    x_poss=[0,l,l+L]

    Gtr=shapes.make_wall_refined_ortho_grid(x_sizes[1],y_sizes[0], dtype=dtype,corner_lower=(x_poss[1],y_poss[1]), corner_upper=(x_poss[2],y_poss[2]),wall_refinement=["-y","+y","-x"],base=base,function=func)
    Gb=shapes.make_wall_refined_ortho_grid(x_sizes[1],y_sizes[1], dtype=dtype,corner_lower=(x_poss[1],y_poss[0]), corner_upper=(x_poss[2],y_poss[1]),wall_refinement=["-y","+y","-x"],base=base,function=func)

    #make the inlet grid with fixed refinement
    # Gtl=shapes.make_wall_refined_ortho_grid(x_sizes[0],y_sizes[0], dtype=dtype,corner_lower=(x_poss[0],y_poss[1]), corner_upper=(x_poss[1],y_poss[2]),wall_refinement=["-y","+y","+x"],base=base,function=func)
    #make the grid with the same resolution as the connected grid
    Gtl,_=shapes.grid_by_neighbor_exp(Gtr,l,x_sizes[0],y_sizes[0],dtype,(x_poss[0],y_poss[1]),(x_poss[1],y_poss[2]),side="-x")

    #make the buffer grid with fixed resolution and physical length
    Gto,ratio=shapes.grid_by_neighbor_exp(Gtr,buffer_length,buffer_res,y_sizes[0],dtype,(x_poss[2],y_poss[1]),(x_poss[2]+buffer_length,y_poss[2]),side="+x")
    print(f"Ratio for neighbor grids:{ratio}")
    Gbo,_=shapes.grid_by_neighbor_exp(Gb,buffer_length,buffer_res,y_sizes[1],dtype,(x_poss[2],y_poss[0]),(x_poss[2]+buffer_length,y_poss[1]),side="+x")

    #make the buffer grid with fixed resolution and mesh size
    # Gto,l1=grid_by_neighbor_linear(Gtr,buffer_res,y_sizes[0],dtype=dtype,corner_lower=(x_poss[2],y_poss[1]), corner_upper=(x_poss[2]+buffer_length,y_poss[2]),side="+x")
    # Gbo,l2=grid_by_neighbor_linear(Gb,buffer_res,y_sizes[1],dtype=dtype,corner_lower=(x_poss[2],y_poss[0]), corner_upper=(x_poss[2]+buffer_length,y_poss[1]),side="+x")

    return [Gtl,Gtr,Gb,Gto,Gbo]

def makeBFSdomain_BufferBlock(l,L,H,h,base,in_vel,in_var,res_x,res_y,dims,viscosity,dtype,func,buffer_list=None,vis_ratio=1,res_l=64,z=None,res_z=None,z_closed=False):
    res_h=int(res_y*h/H)
    grids=make5BFSGrid(l=l,L=L,H=H,h=h,base=base,res_x=res_x,res_y=res_y,buffer_list=buffer_list,dtype=dtype,func="exp",res_l=res_l)
    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, name="BFSDomain", device=cuda_device, dtype=dtype)
    if dims==3:
        grids= [shapes.extrude_grid_z(grid, res_z=res_z,end_z=z, weights_z=None, base=1.0) for grid in grids]
    grids = [_.to(cuda_device).contiguous() for _ in grids]
    for t_idx, coords in enumerate(grids):
        block = domain.CreateBlock(vertexCoordinates=coords, name="BFS%d"%t_idx)
        block.CloseAllBoundaries()
    blocks = domain.getBlocks()
    Btl,Btr,Bb,Bto,Bbo= blocks
    layout=[[-1,2,4],[0,1,3]]
    axes_x = ["-y"] if dims==2 else ["-y", "-z"]
    axes_y = ["-x"] if dims==2 else ["-z", "-x"]
    Btl.ConnectBlock("+x",Btr,"-x",*axes_x)
    Btr.ConnectBlock("-y",Bb,"+y",*axes_y)
    Btr.ConnectBlock("+x",Bto,"-x",*axes_x)
    Bb.ConnectBlock("+x",Bbo,"-x",*axes_x)
    Bto.ConnectBlock("-y",Bbo,"+y",*axes_y)
    # Bb.ConnectBlock("+y",Btr,"-y",*axes_y)

    #3D setting
    if dims==3 and not z_closed:
        for block in blocks:
            block.MakePeriodic("z")
    
    #viscosity
    out_damp_strength=vis_ratio
    damp_start=0
    out_blocks=[Bto,Bbo]
    for out_block in out_blocks:
        visc_parts = []
        viscosity=viscosity.to(cuda_device)
        block_size = out_block.getSizes()
        damp_end=block_size.x//2
        buffer_size=[damp_start,damp_end-damp_start,block_size.x-damp_end]
        visc_parts.append(torch.ones([1,1,block_size.y, buffer_size[0]], dtype=dtype, device=cuda_device)*viscosity)
        visc_parts.append((viscosity*torch.pow(
                        torch.tensor([out_damp_strength], dtype=dtype, device=cuda_device).expand(buffer_size[1]),
                        torch.linspace(0, 1, buffer_size[1], dtype=dtype, device=cuda_device)
                    )).reshape([1,1,1,buffer_size[1]]).expand(-1,-1,block_size.y,-1)
                )
        visc_parts.append(torch.ones([1,1,block_size.y, buffer_size[2]], dtype=dtype, device=cuda_device)*viscosity*out_damp_strength)
        visc = torch.cat(visc_parts, dim=-1).contiguous()
        if dims==3:
            visc = visc.unsqueeze(2).expand(-1,-1,block_size.z,-1,-1)
            visc = visc.contiguous()
        out_block.setViscosity(visc)

    #inlet
    bdims=dims-1
    in_shape=[res_z,res_h] if dims==3 else [res_h]
    out_shape=[res_z,res_y] if dims==3 else [res_y]
    inlet_coords = shapes.get_boundary_coords(shapes.coords_to_center_coords(grids[0]),0) - (H-h) # get the y coords of inlet face, firstly get the coords of the center, then get the coords of the boundary, then subtract the height of the obstacle
    inflow_init = 6 * in_vel * (inlet_coords / h) * (1 - inlet_coords / h)
    if dims==3:
        inflow_init=inflow_init.unsqueeze(0).repeat(res_z, 1)
        outflow_init=outflow_init.unsqueeze(0).repeat(res_z, 1)
    inflow = (torch.reshape(inflow_init, [1,1]+in_shape+[1]))
    inflow = torch.cat([inflow]+[torch.zeros_like(inflow)]*bdims, axis=1).cuda()
    inflow_scalar_mid = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
    inflow_scalar_mid = (torch.reshape(inflow_scalar_mid, [1,1]+in_shape+[1]))
    Btl.getBoundary("-x").setVelocity(inflow)
    Btl.getBoundary("-x").setPassiveScalar(inflow_scalar_mid.to(cuda_device))

    #outlet
    # max_in = torch.max(inflow)
    max_in = torch.mean(inflow) * (h/H)
    outlet_coords = torch.cat([shapes.get_boundary_coords(shapes.coords_to_center_coords(grids[3]),1), shapes.get_boundary_coords(shapes.coords_to_center_coords(grids[4]),1)]) # get the y coords of outlet face
    outflow_init = 6 * in_vel * (h / H) * (outlet_coords / H) * (1 - outlet_coords / H)
    outflow = torch.reshape(outflow_init, [1,1]+out_shape+[1])
    outflow = torch.cat([outflow]+[torch.zeros_like(outflow)]*bdims, axis=1).cuda()
    outflow_blocks =[_.cuda() for _ in torch.split(outflow, [res_h,res_y-res_h], dim=-2)]
    outflow_blocks=[block.contiguous() for block in outflow_blocks]
    Bto.getBoundary("+x").setVelocity(outflow_blocks[0])
    Bbo.getBoundary("+x").setVelocity(outflow_blocks[1])
    # Bto.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
    # Bbo.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
    Bto.getBoundary("+x").CreatePassiveScalar(False)
    Bbo.getBoundary("+x").CreatePassiveScalar(False)
    char_vel = torch.tensor([[max_in]+[0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    out_bound_indices = [domain.getBlocks().index(Bto), domain.getBlocks().index(Bbo)]
    
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())
    out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
    prep_fn_static = lambda it, dt: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, torch.zeros_like(char_vel), dt.cuda())
    prep_fn_static(0, torch.ones([1], dtype=dtype, device=cuda_device))
    domain.PrepareSolve()
    BFS_prep_fn={"PRE": [prep_fn]}

    return domain,layout,BFS_prep_fn,grids

def get_outlet_grid(connected_grid, res_x, res_y, base, corner_lower, corner_upper,func):
    """
    This function is to generate the buffer grid for outlet. Firstly a non-smooth buffer grid will be created then it will be amended to a smooth transition based on the length of the initial buffer grid and connected grid in physical space.
        - prev_grid: the grid that is connected to the buffer grid
        - res: the resolution of the buffer grid
        - base: the base of the exponential function
        - corner_lower: the lower corner of the buffer grid
        - corner_upper: the upper corner of the buffer grid
        - func: the function that will be used to generate the buffer grid
    """
    prev_grid_phy = connected_grid[0, 0, -1, :][-1] - connected_grid[0, 0, -1, :][-2]
    curr_outlet_grid = shapes.make_wall_refined_ortho_grid(res_x, res_y, dtype=dtype, corner_lower=corner_lower, corner_upper=corner_upper, wall_refinement=["-x"], base=base, function=func)
    curr_grid_phy = curr_outlet_grid[0, 0, -1, :][1] - curr_outlet_grid[0, 0, -1, :][0]
    ratio = prev_grid_phy / curr_grid_phy
    l_curr = ratio * (corner_upper[0] - corner_lower[0])
    amended_grid = shapes.make_wall_refined_ortho_grid(res_x, res_y, dtype=dtype, corner_lower=corner_lower, corner_upper=(corner_lower[0] + l_curr, corner_upper[1]), wall_refinement=["-x"], base=base, function=func)
    return amended_grid,l_curr


def add_disturbance(res_y: int, res_x: int, range: int, simulation_dtype,direction: str = "y") -> torch.Tensor:
    """
    This function is for adding a disturbance (sin) to the velocity field.
    res_y: the resolution in y direction
    res_x: the resolution in x direction
    range: the range of the disturbance
    direction: the direction of the disturbance
    """
    # Minimum number of samples for the sine wave and amplitude scaling
    min_samples = 30
    min_amplitude = 5  # Scale up the disturbance for smaller ranges

    velocity = torch.zeros(size=[1, 2, res_y, res_x], dtype=simulation_dtype, device=cuda_device)
    # range=2*range
    if direction == "y":
        # Calculate the number of samples for the sine wave, ensuring at least min_samples are used
        num_samples = max(min_samples, 2 * range)
        # Adjust amplitude based on range
        amplitude = max(min_amplitude, 10 * (range / min_samples))
        # Create sine wave
        sine_wave = amplitude * torch.sin(torch.linspace(0, 2 * np.pi, num_samples)).reshape(1, 1, num_samples, 1)
        # Insert the sine wave into the velocity tensor
        velocity[0, 1, res_y//2-range:res_y//2+range, :] = sine_wave[:, :, :2*range]

    if direction == "x":
        # Similar to 'y' but potentially with different amplitude adjustments
        num_samples = max(min_samples, 2 * range)
        amplitude = max(min_amplitude, 10 * (range / min_samples))
        sine_wave = amplitude * torch.sin(torch.linspace(0, 2 * np.pi, num_samples)).reshape(1, 1, 1, num_samples)
        velocity[0, 0, :, res_x//2-range:res_x//2+range] = sine_wave[:, :, :, :2*range]

    return velocity


def make8BlockChannelFlowSetup(x:int, y:int, z:int, x_in:int, y_in:int, x_pos:int, in_vel:float, in_var:float=0.4, closed_y=False, closed_z=False, viscosity=0.0, scale:float=None,base=None, denser=1.0,dtype=torch.float32,func="exp",disturbance=True) -> PISOtorch.Domain:
    """
    Configures 8 blocks arranged and connected as a "ring", s.t. the missing center block creates an obstacle via closed bounds

    Args:
    - x (int): Total length (stream-wise) of the channel
    - y (int): Total height of the channel, normal to the stream.
    - z (int, optional): Total depth of the channel; set to None for 2D simulations.
    - x_in (int): Width of the obstacle along the x-axis. (in cells)
    - y_in (int): Height of the obstacle along the y-axis. (in cells)
    - x_pos (int): Distance from the inflow to the obstacle, in cells; the obstacle is centered along the y-axis.
    - in_vel (float): Magnitude of the inflow velocity.
    - in_var (float): Variance of the Gaussian inflow profile, applicable when using closed boundaries.
    - closed_y (bool): Specifies whether the upper and lower bounds of the channel are closed (if False, they are periodic).
    - closed_z (bool): Specifies whether the z-bounds of the channel are closed (if False, they are periodic).
    - viscosity (float): Viscosity of the fluid.
    - scale (float, optional): Scaling factor for the cell size, used for normalization across different resolutions; set to None for untransformed grid.
    """
    dims = 2 if z is None else 3
    if denser>1:
        x=x+int((denser-1)*x_in)
        y=y+int((denser-1)*y_in)
        y_in=int(denser*y_in)
        x_in=int(denser*x_in)

    assert x>=(x_in-2)
    assert y>=(y_in-2)
    has_transformation = scale is not None
    vel = [in_vel] + [0]*(dims-1)
    
    # make the grids
    x_sizes = [x_pos, x_in, x - (x_pos+x_in)]
    y_sizes = [(y-y_in)//2, y_in, (y-y_in) - (y-y_in)//2]
    # x_poss = [-x_sizes[0]-(x_sizes[1]/2), -x_sizes[1]/2, x_sizes[1]/2, x_sizes[1]/2+x_sizes[2]]
    # y_poss = [-y_sizes[0]-(y_sizes[1]/2), -y_sizes[1]/2, y_sizes[1]/2, y_sizes[1]/2+y_sizes[2]]
    # make the grids
    x_poss = [-x_sizes[0]-(x_sizes[1]/(2*denser)), -x_sizes[1]/(2*denser), x_sizes[1]/(2*denser), x_sizes[1]/(2*denser)+x_sizes[2]]
    y_poss = [-y_sizes[0]-(y_sizes[1]/(2*denser)), -y_sizes[1]/(2*denser), y_sizes[1]/(2*denser), y_sizes[1]/(2*denser)+y_sizes[2]]
    if scale is not None:
        x_poss = [_*scale for _ in x_poss]
        y_poss = [_*scale for _ in y_poss]
    # make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0,0), corner_upper=(1,1), wall_refinement=[], base=1.05, dtype=torch.float32)
    Gbl = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[0]), corner_upper=(x_poss[1],y_poss[1]),wall_refinement=["+x","+y"],base=base,function=func)
    Gbm = shapes.make_wall_refined_ortho_grid(x_sizes[1], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[1],y_poss[0]), corner_upper=(x_poss[2],y_poss[1]),wall_refinement=["+y","+x","-x"],base=base,function=func)
    Gbr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[0]), corner_upper=(x_poss[3],y_poss[1]),wall_refinement=["+y","-x"],base=base,function=func)
    
    Gml = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[1], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[1]), corner_upper=(x_poss[1],y_poss[2]),wall_refinement=["+x","-y","+y"],base=base,function=func)
    #Gmm = obstacle
    Gmr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[1], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[1]), corner_upper=(x_poss[3],y_poss[2]),wall_refinement=["-x","-y","+y"],base=base,function=func)
        
    Gtl = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[2]), corner_upper=(x_poss[1],y_poss[3]),wall_refinement=["-y","+x"],base=base,function=func)
    Gtm = shapes.make_wall_refined_ortho_grid(x_sizes[1], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[1],y_poss[2]), corner_upper=(x_poss[2],y_poss[3]),wall_refinement=["-y","+x","-x"],base=base,function=func)
    Gtr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[2]), corner_upper=(x_poss[3],y_poss[3]),wall_refinement=["-y","-x"],base=base,function=func)
    
    grids = [Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr]
    # Vmr=add_disturbance(Gmr.shape[-2]-1,Gmr.shape[-1]-1,Gmr.shape[-2]//2,direction="y")

    if dims==3:
        grids = [shapes.extrude_grid_z(grid, z, end_z=z*scale if scale is not None else z, weights_z=None, exp_base=1.05) for grid in grids]
        Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr = grids
    
    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, name="Domain8BlockChannelFlow", device=cuda_device, dtype=dtype)
    
    
    # make the blocks
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockTopLeft")
    Btm = domain.CreateBlock(vertexCoordinates=Gtm.to(cuda_device), name="BlockTopMiddle")
    Btr = domain.CreateBlock(vertexCoordinates=Gtr.to(cuda_device), name="BlockTopRight")
    Bml = domain.CreateBlock(vertexCoordinates=Gml.to(cuda_device), name="BlockMiddleLeft")
    # Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), name="BlockMiddleRight")
    if disturbance:
        Vmr=add_disturbance(Gmr.shape[-2]-1,Gmr.shape[-1]-1,Gmr.shape[-2]//int(denser*4),dtype,direction="y")
        Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), velocity=Vmr,name="BlockMiddleRight")
    else:
        Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), name="BlockMiddleRight") 

    # Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), velocity=Vmr,name="BlockMiddleRight")

    Bbl = domain.CreateBlock(vertexCoordinates=Gbl.to(cuda_device), name="BlockBotLeft")
    Bbm = domain.CreateBlock(vertexCoordinates=Gbm.to(cuda_device), name="BlockBotMiddle")
    Bbr = domain.CreateBlock(vertexCoordinates=Gbr.to(cuda_device), name="BlockBotRight")

    blocks = [Btl, Btm, Btr, Bml, Bmr, Bbl, Bbm, Bbr]
    layout = [[5,6,7],[3,-1,4],[0,1,2]] # used by output formatting
    # layout=[[0, 1, 2], [3, -1, 4], [5, 6, 7]]

    # set boundaries
    # close obstacle
    Btm.CloseBoundary("-y")
    Bml.CloseBoundary("+x")
    Bmr.CloseBoundary("-x")
    Bbm.CloseBoundary("+y")
    if not closed_y:
        # connect to be periodic
        axes = ["-x"] if dims==2 else ["-z", "-x"]
        Btl.ConnectBlock("+y", Bbl, "-y", *axes)
        Btm.ConnectBlock("+y", Bbm, "-y", *axes)
        Btr.ConnectBlock("+y", Bbr, "-y", *axes)
    
    if dims==3 and closed_z:
        for block in blocks:
            block.CloseBoundary("-z") # also closes +z from default periodic
            #block.CloseBoundary("+z")
    
    # connect 8-block ring
    axes_x = ["-y"] if dims==2 else ["-y", "-z"]
    axes_y = ["-x"] if dims==2 else ["-z", "-x"]
    Btl.ConnectBlock("+x", Btm, "-x", *axes_x)
    Btm.ConnectBlock("+x", Btr, "-x", *axes_x)

    Btr.ConnectBlock("-y", Bmr, "+y", *axes_y)
    Bmr.ConnectBlock("-y", Bbr, "+y", *axes_y)
    
    Bbl.ConnectBlock("+x", Bbm, "-x", *axes_x)
    Bbm.ConnectBlock("+x", Bbr, "-x", *axes_x)

    Btl.ConnectBlock("-y", Bml, "+y", *axes_y)
    Bml.ConnectBlock("-y", Bbl, "+y", *axes_y)

    # on block connection specification:
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    # the "axes"-parameter is a bit tricky: it is based on the value o f"side of block_from to connect"
    # and goes over the remaining axes by increasing index, wrapping back to 0 (x) if necessary.
    # It specifies the other axis to connect to. usually this will be simply increasing to keep the setup consistent.
    # The sign indicates whether this connection axis should the inverted (+) or not (-)

    # PISOtorch.ConnectBlocks sets a valid 2-way connection
    
    # specify the inflow
    in_shape = [y] if dims==2 else [z,y]
    bdims = dims-1
    if closed_y:
        # when the boundaries normal to the inflow are closed we use a gauss profile
        inflow = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
        inflow = (torch.reshape(inflow, [1,1]+in_shape+[1])*in_vel)
    else:
        # othewise the inflow is constant
        inflow = torch.ones([1,1]+in_shape+[1], dtype=dtype, device=cpu_device)*in_vel
    mean_in = torch.mean(inflow)
    max_in = torch.max(inflow)
    char_vel = torch.tensor([[max_in]+[0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    inflow = torch.cat([inflow]+[torch.zeros_like(inflow)]*bdims, axis=1)
    inflow_blocks = [_.cuda() for _ in torch.split(inflow, y_sizes, dim=-2)]
    
    Btl.getBoundary("-x").setVelocity(inflow_blocks[2])
    Bml.getBoundary("-x").setVelocity(inflow_blocks[1])
    Bbl.getBoundary("-x").setVelocity(inflow_blocks[0])
    
    in_shape = [y_in] if dims==2 else [z,y_in]
    inflow_scalar_mid = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
    inflow_scalar_mid = (torch.reshape(inflow_scalar_mid, [1,1]+in_shape+[1]))
    Bml.getBoundary("-x").setPassiveScalar(inflow_scalar_mid.to(cuda_device))
    
    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.reshape(torch.FloatTensor([mean_in.cpu(),0] if dims==2 else [mean_in.cpu(),0,0]), [1,dims]+[1]*dims).cuda()
    # print(out_vel.item())
    
    Btr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[2]) * out_vel)
    Bmr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[1]) * out_vel)
    Bbr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[0]) * out_vel)

    # create varying passive scalar for outflow
    Btr.getBoundary("+x").CreatePassiveScalar(False)
    Bmr.getBoundary("+x").CreatePassiveScalar(False)
    Bbr.getBoundary("+x").CreatePassiveScalar(False)

    # we use the prep_fn to pass the update of the outflow boundaries
    out_bound_indices = [domain.getBlocks().index(Btr), domain.getBlocks().index(Bmr), domain.getBlocks().index(Bbr)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())
    
    out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
    prep_fn_static = lambda it, dt: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, torch.zeros_like(char_vel), dt.cuda())
    prep_fn_static(0, torch.ones([1], dtype=dtype, device=cuda_device))
    domain.PrepareSolve()

    return domain, {"PRE": [prep_fn]}, layout,grids