#from local version 0916
import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import pickle
import os
import re
from lib.util.output import _resample_block_data
from lib.data.resample import sample_multi_coords_from_uniform_grid
cuda_device = torch.device("cuda")
from solvers.solver_2d import PISOtorch_simulation # helper for PISO loop
import psutil
import PISOtorch

def Get_in_out_bound(domain,inlet_idnex,outlet_index):
    inlet={i:domain.getBlock(i).getBoundary("-x").getVelocityVarying().detach().clone() for i in inlet_idnex}
    outlet={i:domain.getBlock(i).getBoundary("+x").getVelocityVarying().detach().clone() for i in outlet_index}
    boundary={"inlet":inlet,"outlet":outlet}
    return boundary

def save_global_stats(filename, stats):
    """
    Saves the global statistics to a file.

    Args:
    filename (str): The name of the file where the stats will be saved.
    stats (dict): A dictionary containing the global statistics.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(stats, file)

    print("Saved to {}".format(filename))


def calculate_transform_stats(solver_dict, use_max=True):
    trasnsform_list = [solver_dict[k][5] for k in solver_dict.keys()]
    target_size = None
    # get the target size
    for item in trasnsform_list:
        current_size = item.size(2), item.size(3)
        if target_size is None:
            target_size = current_size
        else:
            if use_max:
                target_size = max(target_size, current_size, key=lambda x: x[0] * x[1])
            else:
                target_size = min(target_size, current_size, key=lambda x: x[0] * x[1])
    # resize all the trasnform matrices
    for (i,item) in enumerate(trasnsform_list):
        trasnsform_list[i]=F.interpolate(item, size=(target_size))
    # concatenate all the trasnform matrices
    all_trasnsform = torch.cat(trasnsform_list,dim=0)
    mean=all_trasnsform.mean(dim=[0, 2, 3], keepdim=True)
    std=all_trasnsform.std(dim=[0, 2, 3], keepdim=True)
    return mean, std

    
def resample_data(data_list,grids,res_x,res_y,fill_max_steps=0):
    """
    This is for resample the data_list into uniform grid with res_x and res_y based on the grids
    Parameters:
    - data_list: list of data from blocks
    - grids: list of grid which is from the refined grid
    - res_x: resolution in x direction
    - res_y: resolution in y direction
    Output:
    - resampled_data: resampled data in the shape of [batch, channels, res_y, res_x]
    """
    #make sure it's in a contiguous memeory space
    data_list=[item.contiguous() for item in data_list]
    resampled_data=_resample_block_data(data_list,grids,[res_x,res_y],ndims=2,fill_max_steps=fill_max_steps)
    return resampled_data


def pad_to_match(block: torch.Tensor, target_shape: Tuple[int]) -> torch.Tensor:
    """
    Pad the block to match the target shape using PyTorch.

    :param block: The block to be padded (PyTorch tensor).
    :param target_shape: The target shape to pad the block to.
    :return: Padded block (PyTorch tensor).
    """
    padding = [(0, max(0, t - s)) for s, t in zip(block.shape, target_shape)]
    # Convert padding format for F.pad (expects a flattened list)
    padding = [val for sublist in reversed(padding) for val in sublist]
    return F.pad(block, padding, mode='constant', value=0)

def calculate_max_sizes(blocks: List, layout: List[List[int]], is_3d: bool) -> Tuple[List[int], List[int], int]:
    """
    Calculate the maximum sizes for rows, columns, and depth.

    :param blocks: A list of PISOtorch.Block objects.
    :param layout: The layout of the blocks.
    :param is_3d: Boolean indicating if the input is 3D.
    :return: Tuple containing max row sizes, max column sizes, and max depth size.
    """
    if is_3d:
        max_row_size = [max(blocks[index].velocity.shape[3] for index in row if index != -1) for row in layout]
        max_col_size = [max(blocks[row[col_idx]].velocity.shape[4] 
                            for row in layout if col_idx < len(row) and row[col_idx] != -1) 
                        for col_idx in range(max(len(row) for row in layout))]
        max_depth_size = max(block.velocity.shape[2] for block in blocks)
    else:
        max_row_size = [max(blocks[index].velocity.shape[2] for index in row if index != -1) for row in layout]
        max_col_size = [max(blocks[row[col_idx]].velocity.shape[3] 
                            for row in layout if col_idx < len(row) and row[col_idx] != -1) 
                        for col_idx in range(max(len(row) for row in layout))]
        max_depth_size = 0  # No depth for 2D case
    
    return max_row_size, max_col_size, max_depth_size

def pad_and_combine_row(blocks: List, row: List[int], max_depth_size: int, max_row_size: List[int], max_col_size: List[int], is_3d: bool) -> Tuple[torch.Tensor, ...]:
    """
    Pad and combine blocks in a single row.

    :param blocks: A list of PISOtorch.Block objects.
    :param row: A list of indices for the current row.
    :param max_depth_size: The maximum depth size.
    :param max_row_size: The maximum row size.
    :param max_col_size: The maximum column size.
    :param is_3d: Boolean indicating if the input is 3D.
    :return: Tuple containing the combined blocks for the row (u_row, v_row, [w_row], pressure_row).
    """
    u_row, v_row, w_row, pressure_row = [], [], [], []

    for col_idx, index in enumerate(row):
        if index == -1:
            dummy_shape = (max_depth_size, max_row_size, max_col_size[col_idx]) if is_3d else (max_row_size, max_col_size[col_idx])
            u_row.append(torch.zeros(dummy_shape, device=blocks[0].velocity.device))
            v_row.append(torch.zeros(dummy_shape, device=blocks[0].velocity.device))
            pressure_row.append(torch.zeros(dummy_shape, device=blocks[0].velocity.device))
            if is_3d:
                w_row.append(torch.zeros(dummy_shape, device=blocks[0].velocity.device))
        else:
            block = blocks[index]
            u_block = pad_to_match(block.velocity[0, 0], (max_depth_size, max_row_size, max_col_size[col_idx]) if is_3d else (max_row_size, max_col_size[col_idx]))
            v_block = pad_to_match(block.velocity[0, 1], (max_depth_size, max_row_size, max_col_size[col_idx]) if is_3d else (max_row_size, max_col_size[col_idx]))
            pressure_block = pad_to_match(block.pressure[0, 0], (max_depth_size, max_row_size, max_col_size[col_idx]) if is_3d else (max_row_size, max_col_size[col_idx]))

            u_row.append(u_block)
            v_row.append(v_block)
            pressure_row.append(pressure_block)
            if is_3d:
                w_block = pad_to_match(block.velocity[0, 2], (max_depth_size, max_row_size, max_col_size[col_idx]))
                w_row.append(w_block)

    if is_3d:
        return torch.cat(u_row, dim=2), torch.cat(v_row, dim=2), torch.cat(w_row, dim=2), torch.cat(pressure_row, dim=2)
    else:
        return torch.cat(u_row, dim=1), torch.cat(v_row, dim=1), torch.cat(pressure_row, dim=1)

def combine_blocks(blocks: List, layout: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine the blocks (PISOtorch.Block) and return the velocity and pressure fields (in torch) for multiple blocks.

    :param blocks: A list of PISOtorch.Block objects.
    :param layout: The layout of the blocks.
    :return: Tuple containing the combined velocity and pressure fields.
    """
    is_3d = len(blocks[0].velocity.shape) == 5
    max_row_size, max_col_size, max_depth_size = calculate_max_sizes(blocks, layout, is_3d)

    u_combined, v_combined, w_combined, pressure_combined = [], [], [], []

    for row_idx, row in enumerate(layout):
        if is_3d:
            u_row, v_row, w_row, pressure_row = pad_and_combine_row(blocks, row, max_depth_size, max_row_size[row_idx], max_col_size, is_3d)
            u_combined.append(u_row)
            v_combined.append(v_row)
            w_combined.append(w_row)
            pressure_combined.append(pressure_row)
        else:
            u_row, v_row, pressure_row = pad_and_combine_row(blocks, row, max_depth_size, max_row_size[row_idx], max_col_size, is_3d)
            u_combined.append(u_row)
            v_combined.append(v_row)
            pressure_combined.append(pressure_row)

    u_torch = torch.cat(u_combined, dim=1 if is_3d else 0)
    v_torch = torch.cat(v_combined, dim=1 if is_3d else 0)
    p_torch = torch.cat(pressure_combined, dim=1 if is_3d else 0)

    if is_3d:
        w_torch = torch.cat(w_combined, dim=1)
        velocity_torch = torch.cat((u_torch.unsqueeze(0).unsqueeze(0),
                                    v_torch.unsqueeze(0).unsqueeze(0),
                                    w_torch.unsqueeze(0).unsqueeze(0)), dim=1)
    else:
        velocity_torch = torch.cat((u_torch.unsqueeze(0).unsqueeze(0),
                                    v_torch.unsqueeze(0).unsqueeze(0)), dim=1)
    
    pressure_torch = p_torch.unsqueeze(0).unsqueeze(0)

    return velocity_torch, pressure_torch

def combine_list(lists, layout):
    device = lists[0].device
    is_1d = lists[0].shape[1] == 1

    max_row_size = [max(lists[index].shape[2] for index in row if index != -1) for row in layout]

    num_columns = max(len(row) for row in layout)
    max_col_size = []
    for col_idx in range(num_columns):
        max_size = 0
        for row in layout:
            if col_idx < len(row) and row[col_idx] != -1:
                max_size = max(max_size, lists[row[col_idx]].shape[3])
        max_col_size.append(max_size)

    u_combined = []
    v_combined = [] if not is_1d else None

    for row_idx, row in enumerate(layout):
        u_row = []
        v_row = [] if not is_1d else None

        for col_idx, index in enumerate(row):
            if index == -1:
                dummy_shape = (max_row_size[row_idx], max_col_size[col_idx])
                u_row.append(torch.zeros(dummy_shape, device=device))
                if not is_1d:
                    v_row.append(torch.zeros(dummy_shape, device=device))
            else:
                block = lists[index]
                u_block = pad_to_match(block[0, 0], (max_row_size[row_idx], max_col_size[col_idx]))
                u_row.append(u_block)

                if not is_1d:
                    v_block = pad_to_match(block[0, 1], (max_row_size[row_idx], max_col_size[col_idx]))
                    v_row.append(v_block)

        u_combined.append(torch.cat(u_row, dim=1))
        if not is_1d:
            v_combined.append(torch.cat(v_row, dim=1))

    u_torch = torch.cat(u_combined, dim=0)

    if is_1d:
        return u_torch.unsqueeze(0).unsqueeze(0)
    else:
        v_torch = torch.cat(v_combined, dim=0)
        return torch.cat((u_torch.unsqueeze(0).unsqueeze(0), v_torch.unsqueeze(0).unsqueeze(0)), dim=1)


def combine_transforms(blocks_data, layout):
    """
    Combine the transform matrices from multiple blocks according to a layout.

    :param blocks_data: A list of blocks, each with a 'transform' tensor attribute.
    :param layout: A matrix defining how the blocks should be spatially arranged.
    :return: Combined transform matrix.
    """
    # Determine the device from the first block's transform for consistency
    device = blocks_data[0].transform.device
    
    # Determine the maximum size for each row
    max_row_size = [max(blocks_data[index].transform.shape[1] for index in row if index != -1) for row in layout]

    # Determine the maximum size for each column
    num_columns = max(len(row) for row in layout)
    max_col_size = []
    for col_idx in range(num_columns):
        max_size = 0
        for row in layout:
            if col_idx < len(row) and row[col_idx] != -1:
                max_size = max(max_size, blocks_data[row[col_idx]].transform.shape[2])
        max_col_size.append(max_size)

    # Loop through the layout and concatenate data
    transform_combined = []

    for row_idx, row in enumerate(layout):
        transform_row = []
        for col_idx, index in enumerate(row):
            if index == -1:
                # Handle empty spaces in the layout
                dummy_shape = (blocks_data[0].transform.shape[0], max_row_size[row_idx], max_col_size[col_idx], 9)
                transform_row.append(torch.zeros(dummy_shape, device=device))
            else:
                block = blocks_data[index]
                target_shape = (block.transform.shape[0], max_row_size[row_idx], max_col_size[col_idx], 9)
                padded_transform = pad_to_match(block.transform, target_shape)
                transform_row.append(padded_transform)
        transform_combined.append(torch.cat(transform_row, dim=2))

    # Combine rows
    transform_torch = torch.cat(transform_combined, dim=1)

    return transform_torch


def Get_velocity(domain,layout):
    blocks_data = [domain.getBlock(blockID) for blockID in range(domain.getNumBlocks())]
    velocity,pressure=combine_blocks(blocks_data, layout)
    return velocity, pressure


def split_into_blocks(velocity_field, layout, block_shape):
    '''
    Split the combined velocity and pressure fields into individual blocks.
    
    Parameters:
        velocity_field (torch.Tensor): Combined velocity field.
        layout (list): Layout of the blocks, should match the dimensions of block_shape.
        block_shape (list): Shape of the individual blocks. The last two dimensions should sum up
                            to match the corresponding dimensions of velocity_field.
    
    Returns:
        list: List of blocks with their respective velocity data.
    '''

    # Validation checks
    if len(layout) != len(block_shape) or any(len(layout[i]) != len(block_shape[i]) for i in range(len(layout))):
        raise ValueError("Layout and block_shape must have the same dimensions.")
    
    total_height = sum(block_shape[i][0][-2] for i in range(len(block_shape)))
    total_width = sum(block_shape[0][i][-1] for i in range(len(block_shape[0])))
    if velocity_field.shape[-2] != total_height or velocity_field.shape[-1] != total_width:
        raise ValueError("Velocity field dimensions do not match the sum of block dimensions.")

    # Initialize the output list with placeholders for each block (None is the obstacle  which not needed)
    max_index = max(max(row) for row in layout if max(row) != -1)
    output_blocks = [None] * (max_index + 1)

    for i in range(len(block_shape)):
        for j in range(len(block_shape[i])):
            if layout[i][j] != -1:  # obstacle are marked with -1
                # Calculate the start and end indices for slicing the tensor
                start_h = sum(block_shape[k][0][-2] for k in range(i))
                start_w = sum(block_shape[i][l][-1] for l in range(j))
                end_h = start_h + block_shape[i][j][-2]
                end_w = start_w + block_shape[i][j][-1]
                if velocity_field.dim() == 4:
                    # Slicing the velocity field to get the required block
                    block = velocity_field[:, :, start_h:end_h, start_w:end_w]
                elif velocity_field.dim() == 5:
                    for top_level_element in block_shape:
                        third_element = top_level_element[0][2]
                        if any(sublist[2] != third_element for sublist in top_level_element):
                            raise ValueError("The third element of each sublist does not match.")
                    block = velocity_field[:, :, :,start_h:end_h, start_w:end_w]
                
                # Assign the block to the correct position in the output list as per layout
                output_blocks[layout[i][j]] = block

    # Remove None entries from the output_blocks list
    output_blocks = [block for block in output_blocks if block is not None]
    return output_blocks

def load_global_stats(filename):
    """
    Loads the global statistics from a file.

    Args:
    filename (str): The name of the file to load the stats from.
    
    Returns:
    dict: A dictionary containing the global statistics.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


def CUDAinfo(device=torch.cuda.current_device()):
    # Get total GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory
    # Convert bytes to GB for easier interpretation
    total_mem_gb = total_mem / (1024**3)

    # Get GPU memory allocated and cached (memory managed by PyTorch's caching allocator)
    allocated_mem = torch.cuda.memory_allocated(device)
    allocated_mem_gb = allocated_mem / (1024**3)
    cached_mem = torch.cuda.memory_reserved(device)
    cached_mem_gb = cached_mem / (1024**3)

    # Get CPU memory usage
    cpu_mem = psutil.virtual_memory()
    total_cpu_mem_gb = cpu_mem.total / (1024**3)
    used_cpu_mem_gb = cpu_mem.used / (1024**3)
    free_cpu_mem_gb = cpu_mem.available / (1024**3)

    print("GPU Memory:")
    # print(f"Total GPU Memory: {total_mem_gb:.2f} GB")
    print(f"Memory Allocated: {allocated_mem_gb:.2f} GB")
    print(f"Memory Cached: {cached_mem_gb:.2f} GB")

    print("\nCPU Memory:")
    # print(f"Total CPU Memory: {total_cpu_mem_gb:.2f} GB")
    print(f"Used CPU Memory: {used_cpu_mem_gb:.2f} GB")
    print(f"Free CPU Memory: {free_cpu_mem_gb:.2f} GB")

# This is for getting velocity from domain (1D) then copy to blocks (2D), the output'shape is similar to the velocity in blocks
def copy_velocity_result_to_blocks(domain,velocity_result=None,spatial_dims=None):
    """
    Copies the velocity result tensor from domain (1D) to blocks, default is advection velocity, but you can also use it for scalars with spatial_dims=1.
    :param domain: The domain object.
    :param velocity_result: The velocity result tensor to copy.
    :param spatial_dims: The number of spatial dimensions.
    :return: A list of velocity tensors corresponding to each block.
    """
    # List to store velocity tensors corresponding to each block
    velocity_block_tensors = []
    
    total_size = domain.getTotalSize()  # The total size across all blocks
    if velocity_result is None:
        velocity_result = domain.velocityResult  
    if spatial_dims is None:
        spatial_dims = domain.getSpatialDims()  
    # Ensure the velocity result tensor is valid
    # assert velocity_result.dim() == 1, "Domain velocity must be a 1D tensor."
    assert velocity_result.size(0) == total_size * spatial_dims, "Domain velocity size mismatch."
    
    for blockID in range(domain.getNumBlocks()):
        block=domain.getBlock(blockID)
        block_size_flat = block.getStrides().w  # Flat size of this block's velocity tensor
        
        # Create an empty tensor to hold the block's reshaped velocity
        block_velocity_tensor = torch.empty((1, spatial_dims, block.getSizes()[1],block.getSizes()[0]), dtype=velocity_result.dtype, device=velocity_result.device)
        
        for dim in range(spatial_dims):
            # Extract the part of the domain velocity result that corresponds to this block
            block_velocity_tensor[0, dim] = velocity_result[block.globalOffset + total_size * dim : block.globalOffset + total_size * dim + block_size_flat].view(block.getSizes()[1],block.getSizes()[0])
        
        # Append the reshaped velocity tensor for this block
        velocity_block_tensors.append(block_velocity_tensor)
    
    return velocity_block_tensors


def transfer_blocks_to_velocity_result(domain, velocity_block_tensors):
    """
    Transfers the velocity data from the blocks back to the 1D velocity result tensor in the domain.
    :param domain: The domain object.
    :param velocity_block_tensors: A list of velocity tensors corresponding to each block.
    :return: The 1D velocity result tensor.
    """
    # Create an empty 1D tensor to store the result
    total_size = domain.getTotalSize()  # The total size across all blocks
    spatial_dims = domain.getSpatialDims()  # Number of spatial dimensions
    dtype = velocity_block_tensors[0].dtype
    device = velocity_block_tensors[0].device
    
    # Initialize the final 1D velocity result tensor
    velocity_result = torch.empty(total_size * spatial_dims, dtype=dtype, device=device)
    
    # Iterate over each block and copy the flattened velocity data back to the 1D tensor
    for blockID, block_velocity_tensor in enumerate(velocity_block_tensors):
        block = domain.getBlock(blockID)
        block_size_flat = block.getStrides().w  # The flattened size of the block's velocity data
        
        for dim in range(spatial_dims):
            # Flatten the block's velocity tensor for this spatial dimension
            flattened_velocity = block_velocity_tensor[0, dim].view(-1)
            
            # Copy the flattened velocity back into the correct position in the 1D velocity_result tensor
            velocity_result[block.globalOffset + total_size * dim : block.globalOffset + total_size * dim + block_size_flat] = flattened_velocity
    
    return velocity_result


def divergence(velocity, reference_domain):
    """
    this function is calculate the divergence of the velocity tensor by solver
    :param velocity: The velocity, can be in blocks (a list), or be a whole tensor (1D tensor).
    :param reference_domain: The reference domain object.
    :return: The divergence of the velocity tensor.
    """
    domain = reference_domain.Copy()
    if isinstance(velocity,list):
        velocity=[item.contiguous() for item in velocity]
        for blockIdx, block in enumerate(domain.getBlocks()):
            block.setVelocity(velocity[blockIdx])
    
    if isinstance(velocity,torch.Tensor):
        assert velocity.dim() ==1,"The global velocity tensor should have 1 dimensions"
        domain.setVelocityResult(velocity)
        domain.CreateVelocityOnBlocks() #to not overwrite tensors of reference_domain
        PISOtorch.CopyVelocityResultToBlocks(domain)
    
    domain.PrepareSolve()
    divergence_global = PISOtorch.ComputeVelocityDivergence(domain)
    results_in_block=copy_velocity_result_to_blocks(domain,velocity_result=divergence_global,spatial_dims=1)
    return results_in_block

def get_block_shape_by_grid(grids,layout):
    """
    This function is for getting the block shapes just from grids and layout, since there may obsticles, so we need some special treatment
    """
    if grids[0].dim() == 4:
        # get the shapes of the grids
        assert all(element.shape[1] == grids[0].shape[1] for element in grids), "The number of channels should be the same"
        max_row_size = [max(grids[index].shape[2]-1 for index in row if index != -1) for row in layout]
        max_col_size=[]
        num_columns = max(len(row) for row in layout)
        for col_idx in range(num_columns):
            max_size = 0
            for row in layout:
                if col_idx < len(row) and row[col_idx] != -1:
                    max_size = max(max_size, grids[row[col_idx]].shape[3]-1)
            max_col_size.append(max_size)
        shape_list=[]
        for row_id,row in enumerate(layout):
            row_list=[]
            for col_id,col in enumerate(row):
                row_list.append([1,2,max_row_size[row_id],max_col_size[col_id]])
            shape_list.append(row_list)
    elif grids[0].dim()==5:
        assert all(element.shape[1] == grids[0].shape[1] for element in grids), "The number of channels should be the same"
        assert all(element.shape[2] == grids[0].shape[2] for element in grids), "Only support the same resolution in z for now"
        # get the shapes of the grids
        max_row_size = [max(grids[index].shape[3]-1 for index in row if index != -1) for row in layout]
        max_col_size=[]
        num_columns = max(len(row) for row in layout)
        for col_idx in range(num_columns):
            max_size = 0
            for row in layout:
                if col_idx < len(row) and row[col_idx] != -1:
                    max_size = max(max_size, grids[row[col_idx]].shape[4]-1)
            max_col_size.append(max_size)
        shape_list=[]
        for row_id,row in enumerate(layout):
            row_list=[]
            for col_id,col in enumerate(row):
                row_list.append([1,3,grids[0].shape[2]-1,max_row_size[row_id],max_col_size[col_id]])
            shape_list.append(row_list)
    return shape_list

def make_divergence_free(domain, input_values):
    # modifies forcings in place!
    input_values=[value.contiguous() for value in input_values]
    assert len(input_values)==domain.getNumBlocks()
    domain = domain.Copy()
    for block, value in zip(domain.getBlocks(), input_values):
        block.clearVelocitySource()
        block.setVelocity(value)
    domain.PrepareSolve()
    sim = PISOtorch_simulation.Simulation(domain=domain, pressure_tol=1e-8)
    sim.make_divergence_free()
    input_values = [block.velocity for block in domain.getBlocks()]
    return input_values


def make_data_divergence_free(data,solver_dict):
    """
    Make all the data in data(list) divergence free
    input:
    data: list of data, each element stands for a time step
    solver_dict: dict, the dict of solver
    """
    for i in range(len(data)):
        index=(tuple(data[i][5][0]),0)
        domain,layout,sim,block_shape=solver_dict[index][0],solver_dict[index][2],solver_dict[index][3],solver_dict[index][4]
        velocity_block=split_into_blocks(data[i][0],layout,block_shape)
        out_block=make_divergence_free(domain,velocity_block)
        data[i][0]=combine_list(out_block,layout)
    print("All input(downsampled) have been made divergence free")

def compute_dp(pressure,transform_matrix):
    if transform_matrix is not None:
        dx, dy = 1, 1  # computational grid
        dp_dx = (pressure[:, :, :, 2:] - pressure[:, :, :, :-2]) / (2 * dx)
        dp_dy = (pressure[:, :, 2:, :] - pressure[:, :, :-2, :]) / (2 * dy)

        # for the walls, set to 0
        dp_dx [:,:,:16,15:17]=0
        dp_dy [:,:,15:17,:16]=0

        # boundary just padding 
        dp_dx = F.pad(dp_dx, (1, 1, 0, 0), mode='replicate')
        dp_dy = F.pad(dp_dy, (0, 0, 1, 1), mode='replicate')
        dp = torch.stack([
            transform_matrix[..., 4] * dp_dx[:, 0, :, :],
            transform_matrix[..., 7] * dp_dy[:, 0, :, :]
        ], dim=1)
    # TODO: Try implement inputting the grids then calculate the dp
    else:
        raise ValueError("transform_matrix is None")
    return dp

def append_dp(data_set,solver_dict):
    #append the dp to the data
    assert all(len(data_set[i]) == 7 for i in range(len(data_set))), "The data should be a list of 7 elements"
    for data in data_set:
        index=(tuple(data[5][0]),0)
        transform_matrix=solver_dict[index][5]
        dp=compute_dp(data[6],transform_matrix)
        data.append(dp)


def largest_zero_rectangle(slice_2d):
    """
    Given a 2D NumPy array 'slice_2d', find the largest rectangular region
    that is entirely zero. Returns (y_min, x_min, y_max, x_max, max_area), then can be used for mask.
    
    If no zero found, returns (None, None, None, None, 0).
    """
    H, W = slice_2d.shape
    
    # 1) Build the 'mask' of 1s where slice_2d == 0
    mask = (slice_2d == 0).astype(np.int32)
    
    # 2) For row-by-row, build up 'height' array
    height = np.zeros(W, dtype=np.int32)
    
    # Variables to track best rectangle (largest area)
    best_area = 0
    best_coords = (None, None, None, None)  # (y_min, y_max,x_min, x_max)
    
    for row in range(H):
        # Update the height array
        for col in range(W):
            if mask[row, col] == 1:
                height[col] += 1
            else:
                height[col] = 0
        
        # Find largest rectangle in histogram "height"
        # We'll store (start_index) in the stack
        stack = []
        
        def pop_stack_and_calc_area(top_idx, cur_idx):
            """
            top_idx: index from stack we're popping
            cur_idx: current column index in the iteration (used as 'right boundary')
            """
            h = height[top_idx]
            # If stack empty after popping, left boundary is -1
            left_boundary = stack[-1] if stack else -1
            # Right boundary is cur_idx - 1
            width = cur_idx - left_boundary - 1
            area = h * width
            
            # Now compute the actual (y_min, y_max, x_min, x_max)
            # 'h' is how many consecutive 1s including 'row', so row_min = row - h + 1
            # row_max = row
            y_max = row +1 
            y_min = row - h + 1
            x_max = cur_idx
            x_min = left_boundary + 1
            
            return area, (y_min, y_max, x_min, x_max)
        
        for cur_col in range(W):
            # While stack top is higher than current height,
            # we pop and compute area.
            while stack and height[stack[-1]] > height[cur_col]:
                top_idx = stack.pop()
                area, coords = pop_stack_and_calc_area(top_idx, cur_col)
                if area > best_area:
                    best_area = area
                    best_coords = coords
            stack.append(cur_col)
        
        # Clear out anything left in stack
        cur_col = W
        while stack:
            top_idx = stack.pop()
            area, coords = pop_stack_and_calc_area(top_idx, cur_col)
            if area > best_area:
                best_area = area
                best_coords = coords
    
    if best_area == 0:
        # means either there are no zeros at all or no rectangular region > 0
        return (None, None, None, None, 0)
    print("In the order -> (y_min,y_max,x_min,x_max,area)")
    return (*best_coords, best_area)

def get_size_from_grid(grid,layout,mask=True):
    grid_all=combine_list(grid,layout)
    grid_all=grid_all.cpu().numpy()

    #delete the repeated points caused by combine the grids
    grid_all=np.delete(grid_all,np.where(np.diff(grid_all[0,1,...],axis=0)[:,-1]==0)[0],axis=2) # y-axis
    grid_all=np.delete(grid_all,np.where(np.diff(grid_all[0,0,...],axis=1)[-1,:]==0)[0],axis=3) # x-axis
    mask_x = largest_zero_rectangle(grid_all[0,0,...])
    mask_y = largest_zero_rectangle(grid_all[0,1,...])
    size_all=torch.stack([torch.diff(grid_all[:,0:1,...],axis=3)[0,...,:-1,:],torch.diff(grid_all[:,1:2,...],axis=2)[0,...,:-1]],axis=1)

    if mask:
        size_all[0,0,mask_x[0]-1:mask_x[1]+1,mask_x[2]-1:mask_x[3]+1]=0
        size_all[0,1,mask_y[0]-1:mask_y[1]+1,mask_y[2]-1:mask_y[3]+1]=0
    return size_all