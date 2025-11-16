# this function is for downsampling domain
import glob, re, os
cudaID = '3'
# cudaID = cudaID or str(get_available_GPU_id(active_mem_threshold=0.8, default=None))
os.environ["CUDA_VISIBLE_DEVICES"] = cudaID
import torch
cuda_device = torch.device("cuda")
import numpy as np
from scipy.interpolate import griddata
from lib.data import shapes
import PISOtorch
from lib.util import domain_io
import config
from DomainManager import BFSDomainManager,KarmanDomainManager
import PISOtorch_simulation

def downsample_field(high_coords, low_coords, high_data):
    """
    Downsamples the high-resolution field data (e.g., velocity or pressure) to a low-resolution grid based on given coordinates.
    (2D has been tested, 3D doesn't TODO 3D test)
    Args:
        high_coords (torch.Tensor): High-resolution cell-center coordinates, shape [1, C, D, H, W] or [1, C, H, W].
        low_coords (torch.Tensor): Low-resolution cell-center coordinates, shape [1, C, D', H', W'] or [1, C, H', W'].
        high_data (torch.Tensor): High-resolution field data, shape [1, C, D, H, W] or [1, C, H, W].
        
    Returns:
        torch.Tensor: Downsampled field data at low-resolution cell centers, shaped to match `low_coords`.
    """
    device = high_coords.device
    dtype = high_data.dtype
    is_3d = (high_coords.dim() == 5 and high_data.dim() == 5)
    channels = high_data.shape[1]
    downsampled_channels = []

    if device is not torch.device("cpu"):
        high_coords = high_coords.cpu()
        low_coords = low_coords.cpu()
        high_data = high_data.cpu()

    # Flatten the high-resolution coordinates (sicpy.griddata requires 2D input)
    if is_3d:
        # 3D Case
        #BCDHW->DHWC
        high_coords_np = high_coords[0].permute(1, 2, 3, 4).reshape(-1, 3).numpy()  # Shape [D*H*W, 3]
        low_coords_np = low_coords[0].permute(1, 2, 3, 4).reshape(-1, 3).numpy()    # Shape [D'*H'*W', 3]
        
        target_shape = low_coords.shape[2:]
        
        # Interpolate each channel independently
        for c in range(channels):
            high_data_np = high_data[0, c].reshape(-1).numpy()  # Shape [D*H*W]
            downsampled_channel = griddata(high_coords_np, high_data_np, low_coords_np, method='cubic').reshape(target_shape)
            downsampled_channels.append(downsampled_channel)
        
    else:
        # 2D Case
        #BCHW->HWC
        high_coords_np = high_coords[0].permute(1, 2, 0).reshape(-1, 2).numpy()    # Shape [H*W, 2]
        low_coords_np = low_coords[0].permute(1, 2, 0).reshape(-1, 2).numpy()      # Shape [H'*W', 2]
        
        target_shape = low_coords.shape[2:]

        for c in range(channels):
            high_data_np = high_data[0, c].reshape(-1).numpy()  # Shape [H*W]
            downsampled_channel = griddata(high_coords_np, high_data_np, low_coords_np, method='cubic').reshape(target_shape)
            downsampled_channels.append(downsampled_channel)

    downsampled_data = torch.tensor(np.stack(downsampled_channels, axis=0),dtype=dtype,device=device).unsqueeze(0)  # Shape [1, C, *target_shape]

    # "Linear" for downsample is fine, but it may cause error when there is a requirement for extropolation with scipy (it just returns nan), like when the low_coords is outside the high_coords
    if torch.isnan(downsampled_data).any():
        raise ValueError("Downsampled data contains NaN values. This may be due to the low-resolution coordinates being outside the high-resolution coordinates.")

    return downsampled_data

def get_boundary_coords(coords, idx):
    """
    Extracts boundary coordinates from a 2D ([1, 2, H, W]) or 3D ([1, 3, D, H, W]) tensor.
    
    Parameters:
        coords (torch.Tensor): The input tensor of shape [1, 2, H, W] for 2D or [1, 3, D, H, W] for 3D.
        idx (int): The index representing the desired boundary. 
                   - For 2D: idx should be 0 to 3.
                   - For 3D: idx should be 0 to 5.
    Returns:
        torch.Tensor: The boundary coordinates tensor.
    """
    if coords.dim() == 4:  # 2D case: [1, 2, H, W]
        if idx == 0:
            return coords[0, 1, :, 0]  # -x 
        elif idx == 1:
            return coords[0, 1, :, -1]  # +x 
        elif idx == 2:
            return coords[0, 0, 0, :]  # -y 
        elif idx == 3:
            return coords[0, 0, -1, :]  # +y boundary
        else:
            raise ValueError("Invalid idx for 2D boundary; should be 0-3.")
    
    elif coords.dim() == 5:  # 3D case: [1, 3, D, H, W]
        if idx == 0:
            return coords[0, :2, :, :, 0]  # -x 
        elif idx == 1:
            return coords[0, :2, :, :, -1]  # +x 
        elif idx == 2:
            return coords[0, [0,2], :, 0, :]  # -y 
        elif idx == 3:
            return coords[0, [0,2], :, -1, :]  # +y 
        elif idx == 4:
            return coords[0, 1:, 0, :, :]  # -z 
        elif idx == 5:
            return coords[0, 1:, -1, :, :]  # +z 
        else:
            raise ValueError("Invalid idx for 3D boundary; should be 0-5.")
    else:
        raise ValueError("coords should be either a 2D or 3D tensor.")

def downsample_boundary_field(high_coords, low_coords, high_bound_data, bound_idx):
    """
    Downsamples the high-resolution boundary data to a low-resolution boundary grid based on given coordinates. Although boundary value is on the face, not cell center. Since for 2D siutation, the boundary is 1D grids, whether it's getting from cell-center or face, it's the same as the other dirrection has been diminished.
    
    so firstly tell which direction it is, then get the coordinates on that dirrection. Since herein the scipy is applied for interpolation, so for 1D case, the interp1d is used. For 3D simulation, it's 2D grids, so very similar to 2D case in downsample_field, just need to take care of the shape of the data.

    Args:
        high_coords (torch.Tensor): High-resolution cell-center coordinates, shape [1, C, D, H, W] or [1, C, H, W].
        low_coords (torch.Tensor): Low-resolution cell-center coordinates for the boundary, shape [1, C, D', H', W'] or [1, C, H', W'].
        high_bound_data (torch.Tensor): High-resolution boundary field data, shape [1, C, D, 1] or [1, C, H, 1].
        bound_idx(int): The index representing the desired boundary.

    Returns:
        torch.Tensor: Downsampled boundary field data at low-resolution boundary cell centers.
    """
    device = high_coords.device
    dtype = high_bound_data.dtype
    high_coords_cpu = high_coords.cpu()
    low_coords_cpu = low_coords.cpu()
    high_bound_data_cpu = high_bound_data.cpu()
    
    #get the boundary coordinates
    high_coord_bound= get_boundary_coords(high_coords_cpu, bound_idx) # Shape [W] or [H] (2D)
    low_coord_bound= get_boundary_coords(low_coords_cpu, bound_idx)
    is_3d = (high_coords_cpu.dim() == 5 and high_bound_data_cpu.dim() == 5)
    channels = high_bound_data_cpu.shape[1]
    downsampled_channels = []

    # 3D need to be tested, the shape has been tested with random data, not with real simulation data
    if is_3d:
        
        high_coords_np = high_coord_bound.permute(1, 2, 0).reshape(-1, 2).numpy()    # Shape [D*H, 2]
        low_coords_np = low_coord_bound.permute(1, 2, 0).reshape(-1, 2).numpy()      # Shape [H'*W', 2]
        target_shape = low_coord_bound.shape[1:]

        for c in range(channels):
            high_data_np = high_bound_data_cpu[0, c].squeeze().reshape(-1).numpy()  # Shape [H*W]
            downsampled_channel = torch.tensor(griddata(high_coords_np, high_data_np, low_coords_np, method='linear')).reshape(target_shape)
            if bound_idx in [0,1]:
                downsampled_channel =downsampled_channel.unsqueeze(-1) 
            elif bound_idx in [2,3]:
                downsampled_channel =downsampled_channel.unsqueeze(-2)
            elif bound_idx in [4,5]:
                downsampled_channel =downsampled_channel.unsqueeze(-3)
            downsampled_channels.append(downsampled_channel)
        
    else:
        # 2D Case, using interp1d for 1D grids
        for c in range(channels):
            high_data_np = high_bound_data_cpu[0, c].squeeze().reshape(-1).numpy()
            # linear_interp=interp1d(y=high_data_np,x=high_coord_bound.numpy(),kind='linear')
            # linear_interp=interp1d(y=high_data_np,x=high_coord_bound.numpy(),kind='cubic')
            # low_data=torch.tensor(linear_interp(low_coord_bound.numpy()))
            # low_data=low_data.reshape(-1,1) if bound_idx in [0,1] else low_data.reshape(1,-1)
            # downsampled_channels.append(low_data)

            # try to use griddata for consistency
            high_coord_bound_np = high_coord_bound.numpy().reshape(-1, 1)
            low_coord_bound_np = low_coord_bound.numpy().reshape(-1, 1)
            # high_data_np = high_bound_data_cpu[0, c].squeeze().reshape(-1).numpy()
            downsampled_boundary_data =torch.tensor(griddata(high_coord_bound_np, high_data_np, low_coord_bound_np, method='cubic'))
            downsampled_channels.append(downsampled_boundary_data)
            
        
    downsampled_data = torch.stack(downsampled_channels, axis=0).unsqueeze(0)  # Shape [1, C, *target_shape]
    
    if torch.isnan(downsampled_data).any():
        raise ValueError("Downsampled data contains NaN values. This may be due to the low-resolution coordinates being outside the high-resolution coordinates.")

    downsampled_data = downsampled_data.to(dtype=dtype, device=device)
    
    return downsampled_data
    
def downsample_domain(low_domain, loaded_domain, only_velocity=False):
    """
    Downsample the high-resolution domain to a low-resolution domain using the given coordinates.
    Args:
        low_domain (PISOtorch.Domain): The low-resolution domain to be downsampled, which need to be initialized in advance.
        loaded_domain (PISOtorch.Domain): The high-resolution domain to be downsampled.
        only_velocity (bool): If True, only downsample velocity (default False, True for calculating loss).
    """
    #clone to avoid changing the original domain as original domain will be used for reference
    high_domain=loaded_domain.Clone()
    low_domain.setViscosity(high_domain.viscosity)

    # scalar viscosity if present
    if high_domain.hasPassiveScalarViscosity():
        low_domain.setScalarViscosity(high_domain.passiveScalarViscosity)

    # downsample the low_domain in place
    for blockId in range(high_domain.getNumBlocks()):
        low_block = low_domain.getBlock(blockId)
        high_block = high_domain.getBlock(blockId)
        # Get coordinates for low and high-resolution grids (center coordinates)
        low_coords = shapes.coords_to_center_coords(low_block.vertexCoordinates)
        high_coords = shapes.coords_to_center_coords(high_block.vertexCoordinates)
        
        # Downsample and set velocity, pressure
        low_block.setVelocity(downsample_field(high_coords, low_coords, high_block.velocity))
        
        if not only_velocity:
            low_block.setPressure(downsample_field(high_coords, low_coords, high_block.pressure))
            
            # Downsample and set scalar if present
            if high_block.hasPassiveScalar():
                low_block.setPassiveScalar(downsample_field(high_coords, low_coords, high_block.passiveScalar))
            
            # Downsample and set velocitySource if present
            if high_block.hasVelocitySource():
                low_block.setVelocitySource(downsample_field(high_coords, low_coords, high_block.velocitySource))
            
            # viscosity cell wise for velocity 
            if high_block.hasViscosity():
                low_block.setViscosity(downsample_field(high_coords, low_coords, high_block.viscosity))

            # viscosity cell wise for passive scalar
            if high_block.hasPassiveScalarViscosity():
                low_block.passiveScalarViscosity(downsample_field(high_coords, low_coords, high_block.passiveScalarViscosity))
            
            # after downsampled all blocks, check or downsample the boundary depending on the boundary type
            for bound_idx in range(low_block.getSpatialDims()*2):
                low_bound=low_block.getBoundary(bound_idx)
                high_bound=high_block.getBoundary(bound_idx)

                # FIXED(DIRICHLET, DIRICHLET_VARYING, wall)
                if low_bound.type==PISOtorch.FIXED:
                    #This is wall
                    if low_bound.isVelocityStatic:
                        if not torch.equal(low_bound.velocity, high_bound.velocity) or low_bound.type!=high_bound.type:
                            raise ValueError("Wall boundary should be the same")
                    
                    # This is DIRICHLET or DIRICHLET_VARYING
                    else:
                        low_bound_velocity=downsample_boundary_field(high_coords, low_coords, high_bound.velocity, bound_idx)
                        low_bound.setVelocity(low_bound_velocity)
                        
                    if low_bound.passiveScalar is not None:
                        if low_bound.isPassiveScalarStatic():
                            if not torch.equal(low_bound.passiveScalar, high_bound.passiveScalar) or low_bound.type!=high_bound.type:
                                raise ValueError("Wall boundary should be the same on passive scalar")
                        else:
                            low_bound_scalar=downsample_boundary_field(high_coords, low_coords, high_bound.passiveScalar, bound_idx)
                            low_bound.setPassiveScalar(low_bound_scalar)
                
                #CONNECTED, no need downsample, but checking the connection
                elif low_bound.type==PISOtorch.CONNECTED:
                    if high_bound.type!=low_bound.type or high_bound.axes!=low_bound.axes:
                        raise ValueError("Connected boundary should be the same")
                
                #PERIODIC, no need downsample, but checking the type
                elif low_bound.type==PISOtorch.PERIODIC:
                    if high_bound.type!=low_bound.type:
                        raise ValueError("Periodic boundary should be the same")
                else:
                    raise TypeError("Unknown boundary type.")
    low_domain.PrepareSolve()

def create_new_paths(path):
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    new_dir = os.path.join(base_dir, "Down_4-2")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    new_path = os.path.join(new_dir, f"Down4-2{filename}")
    return new_path


def downsample_save_domain(run_id, Re, time_range=(0, None)):
    # load domain
    pattern = f"./INC_Data/BFS/Dataset/{run_id}_*/Re{Re}_*/BFSdomain_*.json"
    paths = glob.glob(pattern)
    paths.sort()
    start_timestamp, end_timestamp = time_range
    paths = [path[:-5] for path in paths]  # Remove '.json' extension

    # base_step_pattern = r"Base([0-9\-]+)_Step([0-9]+)"
    base_step_pattern = r"Base([0-9\-]+)_Step([-+]?[0-9]*\.?[0-9]+)"
    id_pattern = r"BFSdomain_(\d+)$"
    geo_groups = {}
    sim=PISOtorch_simulation.Simulation(
            domain=None, time_step=params.time_step, block_layout=None,
            prep_fn=None, substeps=params.substeps, corrector_steps=2,
            pressure_tol=params.pressure_tol, advect_non_ortho_steps=1, differentiable=False,
            pressure_non_ortho_steps=1, pressure_return_best_result=True, velocity_corrector="FD",
            non_orthogonal=False, norm_vel=True, log_dir=None, log_interval=1, save_domain_name=None, stop_fn=None
        )
    filtered_paths = []
    for path in paths:
        id_match = re.search(id_pattern, path)
        if id_match:
            time_stamp = int(id_match.group(1))
            # Check if the timestamp falls within the specified range
            if (start_timestamp is None or time_stamp >= start_timestamp) and (end_timestamp is None or time_stamp <= end_timestamp):
                filtered_paths.append(path)
    for path in filtered_paths:
        match = re.search(base_step_pattern, path)
        id_match = re.search(id_pattern, path)
        if match and id_match:
            base_str = match.group(1)
            s = float(match.group(2))
            if s != params.geo_list[0]:
                raise ValueError(f"Step height mismatch: {s} != {params.geo_list[0]}")
            base = list(map(int, base_str.split("-")))
            if base!=params.base_list[0]:
                raise ValueError(f"Base mismatch: {base} != {params.base_list[0]}")
            config_keys=domain_manager.get_all_keys()[0]
            down_domain = domain_manager.get_config(config_keys).domain
            high_domain=domain_io.load_domain(path, dtype=dtype, device=cuda_device)
            downsample_domain(down_domain,high_domain)
            # sim.domain, sim.block_layout, sim.prep_fn=down_domain, domain_manager.get_config(config_keys).layout, domain_manager.get_config(config_keys).prep_fn
            # sim.make_divergence_free()
            new_path = create_new_paths(path)
            domain_io.save_domain(down_domain, new_path)
            print(f"Downsampled domain saved to {new_path}")

    # for path in paths:
    #     match = re.search(base_step_pattern, path)
    #     id_match = re.search(id_pattern, path)
    #     if match and id_match:
    #         base_str = match.group(1)
    #         s = int(match.group(2))
    #         if s != params["geo_list"][0]:
    #             raise ValueError(f"Step height mismatch: {s} != {params['s']}")
    #         base = list(map(int, base_str.split("-")))
    #         if base!=params["base_list"][0]:
    #             raise ValueError(f"Base mismatch: {base} != {params['base_list']}")
    #         down_domain = domain_manager.get_config(base,s).domain
    #         high_domain=domain_io.load_domain(path, dtype=dtype, device=cuda_device)
    #         downsample_domain(down_domain,high_domain)
    #         new_path = create_new_paths(path)
    #         domain_io.save_domain(down_domain, new_path)
    #         print(f"Downsampled domain saved to {new_path}")

if __name__ == "__main__":
    # run_id="241301-203319"
    # Re=1300
    # time_range=[3901,4500]
    # # time_range=[3300,3900]
    # params= config.BFSSimParams(Re=Re, s=0.875, downsample_factor=4/1)
    # dtype=params.dtype
    # cuda_device=torch.device("cuda")
    # domain_manager = BFSDomainManager(**params.__dict__)
    # downsample_save_domain(run_id, Re, time_range)
    run_id = "241230-234838"
    task = "Karman"
    Re=600
    time_range=[1201,2000]
    params= config.KarmanSimParams(Re=Re, y_in=2.0, downsample_factor=4/2)
    dtype=params.dtype
    domain_manager = KarmanDomainManager(**params.__dict__)
    downsample_save_domain(run_id, Re, time_range)