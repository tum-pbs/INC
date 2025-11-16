import torch
import math
import numpy as np
import scipy.stats as stats
from scipy.spatial.transform import Rotation


def get_grid_coords(size, dtype=torch.int32):
    coords = []
    for dim in size:
        coords.append(torch.range(0,dim-1, dtype=dtype))
    
    coords = torch.meshgrid(*coords, indexing="xy")
    coords = torch.stack(coords)
    
    return coords #CDHW

def get_grid_striped_x(size, dtype=torch.float32):
    coords = get_grid_coords(size, dtype)
    
    #coords_x = coords[0]
    #data = coords_x%2
    
    return coords[0]%2

def make_matrix_rotation_2D(angle, degrees=True):
    if degrees:
        angle = np.deg2rad(angle)
    return np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]],
        dtype=np.float32)

def make_matrix_rotation_3D(rotvec, degrees=True):
    return Rotation.from_rotvec(rotvec, degrees=degrees).as_dcm()

def make_rotation_distance_scaling_fn_sine(angle, r_start, r_end):
    if not r_start>0:
        raise RuntimeError("r_start must be positive.")
    r_half = (r_start+r_end)*0.5
    rad = np.deg2rad(angle)
    def displacement(dist):
        return (1 - (np.cos((dist - r_start)/(r_end - r_start)*2*np.pi)+1)*0.5)
    factor = rad*r_half / displacement(r_half)
    def distance_scaling(dist):
        if dist<r_start or r_end<dist:
            return 0
        else:
            return factor*displacement(dist)/(dist*rad)
    
    return distance_scaling

def make_rotation_distance_scaling_fn_sine_half(angle, r_end):
    
    def distance_scaling(dist):
        if r_end<dist:
            return 0
        else:
            return (np.cos(dist/r_end *np.pi) +1)*0.5
    
    return distance_scaling

def rotate_grid(grid:torch.Tensor, angle:float, axis:list=None, center:str="CENTER", distance_scaling:callable=None):
    # angle: rotation angle in degrees
    # axis: axis to rotate around for 3D
    # center: the position to rotate around. coordinate or string
    # distance_scaling: function to return a weighting factor for axis, depending on a points distance to center
    pass
    assert isinstance(grid, torch.Tensor) and (grid.dim() in [4,5]) and grid.size(1)==(grid.dim()-2), "grid must be a torch tensor with shape NCHW or NCDHW"
    dims = grid.dim()-2
    if dims==3:
        assert isinstance(axis, (list, tuple)) and len(axis)==3, "3D rotation axis is required"
        axis = np.asarray(axis, dtype=np.float32)
        axis_norm = np.linalg.norm(axis)
        if axis_norm<1e-5:
            raise ValueError("rotation axis is too short for normalization")
        axis /= axis_norm
    
    grid = torch.moveaxis(grid, 1, -1)
    grid_size = grid.size()
    grid = torch.reshape(grid, (-1,dims))
    
    if center=="ORIGIN":
        center = [0]*dims
    elif center=="CENTER":
        lower, _ = grid.min(dim=0)
        upper, _ = grid.max(dim=0)
        center = [(l+u)*0.5 for l,u in zip(lower, upper)]
    
    assert isinstance(center, (list, tuple)) and len(center)==dims
    center = torch.tensor(center, device=grid.device, dtype=grid.dtype)
    grid = grid - center # now centered on origin
    
    if distance_scaling is not None:
        distances = torch.linalg.norm(grid, dim=-1, keepdims=False).cpu().numpy()
        angle = [angle*distance_scaling(distance) for distance in distances]
        
        make_matrix_rotation = make_matrix_rotation_2D if dims==2 else lambda angle: make_matrix_rotation_3D(angle*axis)
        rotation_matrices = np.asarray([make_matrix_rotation(a) for a in angle])
        
        rotation_matrices = torch.tensor(rotation_matrices, device=grid.device, dtype=grid.dtype)
        
    else:
        if dims==2:
            rotation_matrix = make_matrix_rotation_2D(angle)
        else:
            rotation_matrix = make_matrix_rotation_3D(angle*axis)
        rotation_matrix = torch.tensor(rotation_matrix, device=grid.device, dtype=grid.dtype)
        rotation_matrices = rotation_matrix.reshape((1,dims,dims)).repeat(grid.size(0),1,1)
    
    
    grid = torch.reshape(grid, (-1,dims,1))
    grid = torch.bmm(rotation_matrices, grid)
    grid = torch.reshape(grid, (-1,dims))
    
    grid = grid + center
    grid = torch.reshape(grid, grid_size)
    grid = torch.moveaxis(grid, -1, 1)
    
    return grid

def get_grid_cube_centered(size, cube_size, dtype=torch.float32):
    data = []
    for dim, cs in zip(size, cube_size):
        assert cs<=dim
        border1 = (dim-cs)//2
        border2 = dim - cs - border1
        x = [0]*border1 + [1]*cs + [0]*border2
        data.append(torch.tensor(x, dtype=dtype))
    
    data = torch.meshgrid(*data)
    data = torch.prod(torch.stack(data), dim=0)
    
    return data

def get_grid_normal_dist(size, mean, var, dtype=torch.float32):
    #https://stackoverflow.com/questions/10138085/how-to-plot-normal-distribution
    data = []
    for dim, m, v in zip(size, mean, var):
        s = math.sqrt(v)
        #x = np.linspace(m-3*s, m+3*s, dim)
        # grid borders -1 to 1, so cell centers have some offset
        cell_coord = (dim/2 - 0.5)/(dim/2)
        x = np.linspace(-cell_coord, cell_coord, dim)
        data.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))
    
    data = torch.meshgrid(*data)
    data = torch.prod(torch.stack(data), dim=0)
    
    return data


def ortho_transform_to_coords(transform, dims):
    # transform shape: NDHWC
    assert transform.shape[-1] == 2*dims*dims+1
    coords = []
    for dim in range(dims): #x,y,z
        scales = transform[...,(dims+1)*dim]
        # padding argument goes in inverse dimension order
        coord = torch.nn.functional.pad(torch.cumsum(scales, dim=-(dim+1)), [0,0]*dim + [1,0], value=0)
        coord = torch.nn.functional.pad(coord, [1,0]*dim + [0,0] + [1,0]*(dims - 1 - dim), mode="replicate")
        coords.append(coord)
    coords = torch.stack(coords, dim=1)
    return coords #NCDHW


def coords_to_center_coords(coords):
    # coords NCDHW
    dims = coords.shape[1]
    if dims==2:
        pool = torch.nn.AvgPool2d(2, stride=1)
    elif dims==3:
        pool = torch.nn.AvgPool3d(2, stride=1)
    else:
        raise ValueError()
    return pool(coords)

def get_grid_normal_dist_from_ortho_transforms(transforms, mean, var, dtype=torch.float32, normalize=True):
    # transforms, Coords: NCDHW
    dims = len(transforms.shape)-2
    vertex_coords = ortho_transform_to_coords(transforms, dims)
    coords = coords_to_center_coords(vertex_coords)
    size = [transforms.shape[-(i+1)] for i in range(dims)]
    data = []
    for dim, (res, m, v) in enumerate(zip(size, mean, var)):
        s = math.sqrt(v)
        #x = np.linspace(m-3*s, m+3*s, dim)
        slicing = tuple([0,dim] + [slice(None) if dim==(dims-1-d) else 0 for d in range(dims)])
        #print(slicing)
        x = coords[slicing].cpu().numpy()
        #print(x)
        if normalize:
            slicing = tuple([0,dim] + [0]*dims)
            dim_min = vertex_coords[slicing].cpu().numpy()
            slicing = tuple([0,dim] + [-1]*dims)
            dim_max = vertex_coords[slicing].cpu().numpy()
            #print(dim_min, dim_max)
            #x = x / x[-1] * 2 -1 # -> [-1,1]
            x = (x-dim_min)/(dim_max - dim_min) *2 -1
            #print(x)
        data.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))
    
    data = data[::-1]
    
    data = torch.meshgrid(*data) # C-DHW
    data = torch.prod(torch.stack(data), dim=0) # DHW
    
    return data

def interpolate_vertices_from_borders_2D(borders, x_weights=None, y_weights=None, dtype=torch.float32):
    # borders: 2D: [-x,+x,-y,+y]
    assert len(borders)==4, "only 2D for now"
    dims=2
    res = [len(borders[0]), len(borders[2])] # y,x
    assert len(borders[1])==res[0]
    assert len(borders[3])==res[1]
    
    borders = [np.asarray(border) for border in borders]

    grid = torch.zeros((1,dims,res[0],res[1]), dtype=dtype)
    # set borders of grid
    # for y_idx in range(0,res[0]):
        # grid[0,0,y_idx, 0] = borders[0][y_idx][0]
        # grid[0,1,y_idx, 0] = borders[0][y_idx][1]
        # grid[0,0,y_idx, -1] = borders[1][y_idx][0]
        # grid[0,1,y_idx, -1] = borders[1][y_idx][1]
    # for x_idx in range(0,res[1]):
        # grid[0,0,0, x_idx] = borders[2][x_idx][0]
        # grid[0,1,0, x_idx] = borders[2][x_idx][1]
        # grid[0,0,-1, x_idx] = borders[3][x_idx][0]
        # grid[0,1,-1, x_idx] = borders[3][x_idx][1]
    
    
    if x_weights is None:
        x_weights = [i/(res[0]-1) for i in range(res[0])]
    else:
        assert len(x_weights) == res[0]
    
    if y_weights is None:
        y_weights = [i/(res[1]-1) for i in range(res[1])]
    else:
        assert len(y_weights) == res[1]
    

    # interpolate inner
    # for y_idx in range(1,res[0]-1):
        # for x_idx in range(1,res[1]-1):
            # y_weight = x_weights[x_idx] * (1 - x_weights[x_idx])
            # y_weight = y_weight**2
            # y_weight_upper = y_weights[y_idx] * y_weight #* 0.5
            # y_weight_lower = (1 - y_weights[y_idx]) * y_weight #* 0.5
            # x_weight = y_weights[y_idx] * (1 - y_weights[y_idx])
            # x_weight = x_weight**2
            # x_weight_upper = x_weights[x_idx] * x_weight #* 0.5
            # x_weight_lower = (1 - x_weights[x_idx]) * x_weight #* 0.5
            
            # #y_weight_upper = y_weight_upper**2
            # #y_weight_lower = y_weight_lower**2
            # #x_weight_upper = x_weight_upper**2
            # #x_weight_lower = x_weight_lower**2
            
            # weight_norm = 1/(y_weight_upper + y_weight_lower + x_weight_upper + x_weight_lower)
            # y_weight_upper *= weight_norm
            # y_weight_lower *= weight_norm
            # x_weight_upper *= weight_norm
            # x_weight_lower *= weight_norm
            
            # grid[0,0,y_idx, x_idx] = borders[0][y_idx][0]*x_weight_lower + borders[1][y_idx][0]*x_weight_upper + borders[2][x_idx][0]*y_weight_lower + borders[3][x_idx][0]*y_weight_upper
            # grid[0,1,y_idx, x_idx] = borders[0][y_idx][1]*x_weight_lower + borders[1][y_idx][1]*x_weight_upper + borders[2][x_idx][1]*y_weight_lower + borders[3][x_idx][1]*y_weight_upper
            
    #for y_idx in range(1,res[0]-1):
    for y_idx in range(res[0]):
        y_weight_upper = x_weights[y_idx]
        y_weight_lower = (1 - x_weights[y_idx])
        x_start = borders[2][0]*y_weight_lower + borders[3][0]*y_weight_upper
        x_end   = borders[2][-1]*y_weight_lower + borders[3][-1]*y_weight_upper
        x_size = x_end - x_start
            
        target_size = borders[1][y_idx] - borders[0][y_idx]
        size_diff = target_size - x_size
        epsilon = 1e-7  # small constant
        size_fac = target_size / (x_size + epsilon)
        
        #for x_idx in range(1,res[1]-1):
        for x_idx in range(res[1]):
            x_val = borders[2][x_idx]*y_weight_lower + borders[3][x_idx]*y_weight_upper
            #x_frac = x_val/x_size
            x_frac = x_idx/(res[1]-1)#y_weights[y_idx]
            #x_val = x_val * (borders[1][y_idx][0] - borders[0][y_idx][0]) + borders[0][y_idx]
            if np.any(np.isclose(x_size,0)):
                x_val = x_val - x_start + size_diff * x_frac + borders[0][y_idx]
            else:
                x_val = (x_val - x_start)*size_fac + borders[0][y_idx]
            
            grid[0,0,y_idx, x_idx] = x_val[0]
            grid[0,1,y_idx, x_idx] = x_val[1]
    
    
    return grid

def _check_weights(weights, res, name="weights"):
    if not (len(weights) == res): raise ValueError("Invalid %s: length must match resolution."%(name,))
    if not (isinstance(weights, (list, tuple)) and all(isinstance(w, (int, float)) for w in weights)):
        raise TypeError("Invalid %s: weights must be a list of float."%(name,))
    if not all((0-1e-5)<=w and w<=(1+1e-5) for w in weights): raise ValueError("Invalid %s: weights must be in [0,1]: %s"%(name,weights))
    if not np.isclose(weights[0], 0): raise ValueError("Invalid %s: start weight must be 0, is %s."%(name,weights[0]))
    if not np.isclose(weights[-1], 1): raise ValueError("Invalid %s: end weight must be 1, is %s."%(name,weights[-1]))
    if not all(weights[i]<weights[i+1] for i in range(len(weights)-1)): raise ValueError("Invalid %s: weights must be strictly increasing."%(name,))

def invert_weights(weights):
    _check_weights(weights, len(weights))
    
    sizes = [weights[i+1] - weights[i] for i in range(len(weights)-1)]
    inv_weights = [0]
    size = 0
    # inverse cumulative sum of the sizes
    for i in range(len(sizes)-1,-1,-1):
        size = size + sizes[i]
        inv_weights.append(size)
    
    return inv_weights

def make_weights_linear(res, base=1, refinement="NONE"):
    #base and refinement are not used only for compatibility with other functions
    return [x/(res) for x in range(res+1)]

def make_weights_exp(res, base, refinement):
    # refinement: "START", "END", "BOTH"
    
    exponents = [e for e in range(res)]
    if refinement=="END":
        exponents.reverse()
    elif refinement=="BOTH":
        exponents = exponents[:res//2] + list(reversed(exponents))[res//2:]
    
    sizes = [base**e for e in exponents]
    total_size = np.sum(sizes)
    weights = [0] + [w/total_size for w in np.cumsum(sizes)]
    
    return weights

def make_weights_tanh(res, base, refinement):
    if base==1:
        return make_weights_linear(res,1,refinement)
    # Define a fixed number of exponents based on base
    else:
        factor = 1 / base
        num_points = 9  # fix the points used to generate the weights, then interpolate into the resolution
        exponents = [e * factor for e in range(1, num_points + 1)]
        
        if refinement == "END":
            exponents.reverse()
        elif refinement == "BOTH":
            exponents = exponents[:num_points // 2] + list(reversed(exponents))[num_points // 2:]

        # Calculate tanh for these exponents
        sizes = [np.tanh(e) for e in exponents]
        total_size = np.sum(sizes)
        # weights = [w / total_size for w in sizes]
        weights = [0] + [w/total_size for w in np.cumsum(sizes)]
        # Interpolate these weights to match the resolution 'res'
        # Create an interpolation function
        base_indices = np.linspace(0, 1, num=num_points+1)  # Points where we know the weights
        res_indices = np.linspace(0, 1, num=res+1)  # Points where we want to know the weights
        interpolated_weights = np.interp(res_indices, base_indices, weights)
        interpolated_weights=[float(w) for w in interpolated_weights]

        return interpolated_weights

def make_weights_exponent(res, base,  refinement="NONE"):
    if base==1:
        return make_weights_linear(res,1,refinement)
    else:
        # this parameter maters, don't change it
        num_points=1000
        r = base ** (1 / (num_points - 1))
        weights = r ** np.arange(num_points) 
        total_size = np.sum(weights)
        weights = [0] + [w/total_size for w in np.cumsum(weights)]
        diff=np.diff(weights)

        if refinement == "END":
            weights=np.cumsum(np.insert(diff[::-1],0,0))

        elif refinement == "BOTH":
            k = 2 * np.log(4 * base) / num_points
            n_values = np.arange(num_points+1)
            weights = 1 / (1 + np.exp(-k * (n_values - num_points / 2)))
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        # Interpolate to fit the desired resolution
        base_indices = np.linspace(0, 1, num=num_points+1)
        res_indices = np.linspace(0, 1, num=res+1)
        interpolated_weights = np.interp(res_indices, base_indices, weights)
        interpolated_weights = [float(w) for w in interpolated_weights]

        return interpolated_weights
    
def make_weights_sin(res, base, refinement):
    # the base and refinement parameters are not used, only for compatibility with other functions
    if base == 1:
        return make_weights_linear(res,1,refinement)

    else:
        # Define a range of exponents that fit within the sine wave period of -π to π
        num_points = 100
        exponents = np.linspace(0.01*np.pi, base*1.99*np.pi, num=num_points)

        # Calculate sine for these exponents
        sizes = np.absolute(np.sin(exponents))
        total_size = np.sum(sizes)
        # total_size = sizes

        # Normalize and accumulate sizes
        weights = [0] + [w / total_size for w in np.cumsum(sizes)]

        # Interpolate these weights to match the resolution 'res'
        base_indices = np.linspace(0, 1, num=num_points+1)  # Points where we know the weights
        res_indices = np.linspace(0, 1, num=res+1)  # Points where we want to know the weights
        interpolated_weights = np.interp(res_indices, base_indices, weights)
        interpolated_weights = [float(w) for w in interpolated_weights]

        return interpolated_weights

# this function with shift and scale to ensure that the sine function values stay within 0 to 1 and to enhance the sharpness of the transition, but it's too sharp to use

# def make_weights_sin(res, base, refinement):
#     if base == 1:
#         return make_weights_linear(res)  # assuming make_weights_linear is defined elsewhere

#     else:
#         # Increase the frequency by multiplying by a factor based on 'base'
#         frequency_multiplier = base

#         # Define a range of exponents that fit within the sine wave period of -π to π
#         num_points = 100
#         power=2
#         exponents = np.linspace(0*np.pi, 2*np.pi * frequency_multiplier, num=num_points)

#         # Calculate sine, shift, scale, and raise to a power for sharper transitions
#         sizes = ((np.sin(exponents) + 1) / 2) ** power

#         # Normalize and accumulate sizes
#         total_size = np.sum(sizes)
#         weights = [0] + [w / total_size for w in np.cumsum(sizes)]

#         # Interpolate these weights to match the resolution 'res'
#         base_indices = np.linspace(0, 1, num=num_points+1)  # Points where we know the weights
#         res_indices = np.linspace(0, 1, num=res+1)  # Points where we want to know the weights
#         interpolated_weights = np.interp(res_indices, base_indices, weights)
#         interpolated_weights = [float(w) for w in interpolated_weights]

#         return interpolated_weights


#this determines which function has been applied to make the weights
# make_weights_exp=make_weights_exponent
# make_weights_exp=make_weights_sin
# make_weights_exp=make_weights_tanh

def generate_grid_vertices_2D(res, corner_vertices, border_vertices=None, x_weights=None, y_weights=None, dtype=torch.float32):
    # res: grid resolution in [y,x]
    # corner_vertices: [-x-y,+x-y,-x+y,+x+y], (x,y)-tuple
    # border_vertices: [-x,+x,-y,+y], lists of (x,y)-tuple. will be interpolated from corners if None
    # boundaries: list of str, boundaries to generate an additional layer for
    border_to_corners = {0:(0,2),1:(1,3),2:(0,1),3:(2,3)}
    assert isinstance(corner_vertices, (tuple, list))
    assert len(corner_vertices)==4

    if border_vertices is None:
        border_vertices = [None] * 4
    
    if x_weights is None:
        # weights for x-boundaries, so based on y coordinate
        x_weights = [i/(res[0]-1) for i in range(res[0])]
    else:
        _check_weights(x_weights, res[0], "x_weights")
    
    if y_weights is None:
        y_weights = [i/(res[1]-1) for i in range(res[1])]
    else:
        _check_weights(y_weights, res[1], "y_weights")

    for border_idx in range(len(border_vertices)):
        r = res[border_idx//2]
        if border_vertices[border_idx] is None:
            lower_corner = corner_vertices[border_to_corners[border_idx][0]]
            upper_corner = corner_vertices[border_to_corners[border_idx][1]]
            weights = x_weights if border_idx<2 else y_weights
            border_vertices[border_idx] = []
            for idx in range(r):
                weight_upper = weights[idx]
                weight_lower = 1 - weight_upper
                border_vertices[border_idx].append((lower_corner[0]*weight_lower + upper_corner[0]*weight_upper, lower_corner[1]*weight_lower + upper_corner[1]*weight_upper))
        else:
            assert len(border_vertices[border_idx]) == r, "is %d, expected %d"%(len(border_vertices[border_idx]), r)
            # TODO: check that corners match
    
    #print(border_vertices)

    return interpolate_vertices_from_borders_2D(border_vertices, x_weights=x_weights, y_weights=y_weights, dtype=dtype)

def extrapolate_boundary_layers(grid, boundaries=[]):
    # linear extrapolation of the boundaries
    # boundaries: list of tuple: boundary and extrapolation scale [("-x", 0.5),]
    assert isinstance(grid, torch.Tensor) and grid.dim()==4, "grid must be a 2D torch tensor of shape NCHW"
    assert grid.size(-1)>1 and grid.size(-2)>1, "grid must be at least 2x2"
    assert all(bound in ["-x","+x","-y","+y"] and scale>0 for bound, scale in boundaries)
    
    for bound, scale in boundaries:
        if bound=="-x":
            layer_1 = grid[...,:1]
            layer_2 = grid[...,1:2]
        if bound=="+x":
            layer_1 = grid[...,-1:]
            layer_2 = grid[...,-2:-1]
        if bound=="-y":
            layer_1 = grid[...,:1,:]
            layer_2 = grid[...,1:2,:]
        if bound=="+y":
            layer_1 = grid[...,-1:,:]
            layer_2 = grid[...,-2:-1,:]
        
        bound_layer = layer_1 + (layer_1 - layer_2)*scale
        
        if bound[0]=="-":
            grid = [bound_layer, grid]
        else:
            grid = [grid, bound_layer]
        if bound[1]=="x":
            dim = -1
        else:
            dim = -2
        
        grid = torch.cat(grid, dim=dim)
    
    return grid

def get_extrapolated_boundary_layer(grid, bound, scale):
    assert isinstance(grid, torch.Tensor) and grid.dim()==4, "grid must be a 2D torch tensor of shape NCHW"
    assert grid.size(-1)>1 and grid.size(-2)>1, "grid must be at least 2x2"
    assert bound in ["-x","+x","-y","+y"]
    
    
    if bound=="-x":
        layer_1 = grid[...,:1]
        layer_2 = grid[...,1:2]
    if bound=="+x":
        layer_1 = grid[...,-1:]
        layer_2 = grid[...,-2:-1]
    if bound=="-y":
        layer_1 = grid[...,:1,:]
        layer_2 = grid[...,1:2,:]
    if bound=="+y":
        layer_1 = grid[...,-1:,:]
        layer_2 = grid[...,-2:-1,:]
    
    bound_layer = layer_1 + (layer_1 - layer_2)*scale
    
    if bound[0]=="-":
        grid = [bound_layer, layer_1]
    else:
        grid = [layer_1, bound_layer]
    if bound[1]=="x":
        dim = -1
    else:
        dim = -2
    
    return torch.cat(grid, dim=dim)

def growth_factor_from_sizes(res, x, x0, eps=1e-5, print_info_fn=None):
    from scipy.optimize import least_squares
    # res: number of cells
    # x: total size
    # x0: base cell size
    # finds exponent b s.t. x = sum[i=0, res-1](x0 * b**i) = x0*(1-b**res)/(1-b)
    if not (res>0 and x>x0 and x0>0 and eps>0): raise ValueError("invalid input")
    
    res = np.asarray([res])
    x = np.asarray([x])
    x0 = np.asarray([x0])
    
    if print_info_fn: print_info_fn("least squares: res=%.03e, x==%.03e, x0==%.03e, x/x0=%.03e.", res, x, x0, x/x0)
    
    if np.isclose(res, x/x0):
        return 1
    
    def func(b):
        return x0*(1-np.power(b,res))/(1-b) - x
    
    if res<x/x0:
        b0 = 1+eps
        lb = b0
        ub = 2
    else:
        b0 = 1-eps
        lb = 0+eps
        ub = b0
    
    b0 = np.asarray([b0])
    lb = np.asarray([lb])
    ub = np.asarray([ub])
    
    if print_info_fn: print_info_fn("least squares: b0=%.03e, lb==%.03e, ub==%.03e.", b0, lb, ub)
    
    b = least_squares(func, b0, bounds=(lb, ub))
    
    if print_info_fn: print_info_fn("least squares: %s.", b)
    
    return b.x

def make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0,0), corner_upper=(1,1), wall_refinement=[], base=1.0,function="linear", dtype=torch.float32):
    weights_func = {
        "linear": make_weights_linear,
        "exp": make_weights_exponent,
        "sin": make_weights_sin,
        "tanh": make_weights_tanh
    }
    make_weights_func = weights_func.get(function)
    dims=2
    assert isinstance(corner_lower, (list, tuple)) and len(corner_lower)==2
    assert isinstance(corner_upper, (list, tuple)) and len(corner_upper)==2
    corners = [tuple(corner_lower), (corner_upper[0], corner_lower[1]), (corner_lower[0], corner_upper[1]), tuple(corner_upper)]
    #y_weights = [(i/(res_x))**exponent for i in range(res_x+1)]
    #y_weights_inv = [1 - (1 - i/(res_x))**exponent for i in range(res_x+1)]
    #x_weights = [(i/(res_y))**exponent for i in range(res_y+1)]
    #x_weights_inv = [1 - (1 - i/(res_y))**exponent for i in range(res_y+1)]
    
    if not isinstance(base, (list, tuple)):
        base = [base]*dims

    #transformation of block
    y_w = None
    if "-x" in wall_refinement:
        if "+x" in wall_refinement:
            y_w=make_weights_func(res_x, base=base[0], refinement="BOTH") 
            # y_w = make_weights_exp(res_x, base=base[0], refinement="BOTH") # y_weights[:res_x//2] + y_weights_inv[res_x//2:]
        else:
            y_w = make_weights_func(res_x, base=base[0], refinement="START")
            # y_w = make_weights_exp(res_x, base=base[0], refinement="START") # y_weights
    elif "+x" in wall_refinement:
        y_w = make_weights_func(res_x, base=base[0], refinement="END")
        # y_w = make_weights_exp(res_x, base=base[0], refinement="END") # y_weights_inv
    
    x_w = None
    if "-y" in wall_refinement:
        if "+y" in wall_refinement:
            x_w = make_weights_func(res_y, base=base[1], refinement="BOTH") 
            # x_w = make_weights_exp(res_y, base=base[1], refinement="BOTH") # x_weights[:res_x//2] + x_weights_inv[res_x//2:]
        else:
            x_w = make_weights_func(res_y, base=base[1], refinement="START")
            # x_w = make_weights_exp(res_y, base=base[1], refinement="START") # x_weights
    elif "+y" in wall_refinement:
        x_w = make_weights_func(res_y, base=base[1], refinement="END")
        # x_w = make_weights_exp(res_y, base=base[1], refinement="END") # x_weights_inv
    
    grid = generate_grid_vertices_2D([res_y+1,res_x+1],corners, None, x_weights=x_w, y_weights=y_w, dtype=dtype)
    
    return grid

def extrude_grid_z(grid, res_z, start_z=0, end_z=1, weights_z=None, base=1.0):
    # res_z z resolution of the cell grid. coordinates grid will have res_z+1.
    make_weights_func = {
        "linear": make_weights_linear,
        "exp": make_weights_exponent,
        "sin": make_weights_sin,
        "tanh": make_weights_tanh
    }
    assert grid.dim() == 4 and grid.size(1)==2
    res_x = grid.size(-1)-1
    res_y = grid.size(-2)-1

    if isinstance(weights_z, list):
        assert len(weights_z)==(res_z+1)
    elif weights_z is None or weights_z=="LINEAR":
        # weights_z = make_weights_linear(res_z)
        weights_z = make_weights_func.get("linear")(res_z)
    elif weights_z=="EXP" or weights_z=="EXP_BOTH":
        # weights_z = make_weights_exp(res_z, base=exp_base, refinement="BOTH")
        weights_z = make_weights_func.get("exp")(res_z, base=base, refinement="BOTH")
    elif weights_z=="EXP_START":
        # weights_z = make_weights_exp(res_z, base=exp_base, refinement="START")
        weights_z = make_weights_func.get("exp")(res_z, base=base, refinement="START")

    elif weights_z=="EXP_END":
        weights_z = make_weights_func.get("exp")(res_z, base=base, refinement="EXP_END")
        # weights_z = make_weights_exp(res_z, base=exp_base, refinement="EXP_END")
    else:
        raise ValueError("Unknown weights specification")
    
    def lerp(a,b,t):
        return a*(1-t) + b*t
    coords_z = torch.tensor([lerp(start_z, end_z, w) for w in weights_z], device=grid.device, dtype=grid.dtype)

    coords_z = coords_z.reshape((1,1,res_z+1,1,1)).repeat(1,1,1,res_y+1, res_x+1)
    grid = grid.reshape((1,2,1,res_y+1, res_x+1)).repeat(1,1,res_z+1,1,1)
    grid = torch.cat([grid, coords_z], dim=1)

    return grid
    


def make_torus_2D(res:int, r1:float, r2:float, start_angle:float, angle:float, offset=None, dtype=torch.float32):
        # res: x-resolution along angle, y is computed to result in approx. square cells
        # r1: inner radius. r2: outer radius
        # angles in degrees, start_angle=0 is x-axis, angle goes counterclockwise
        # offset: 2-tuple of float (x,y), "CENTER", None
        # returns: grid of coordinate with layout NCHW with C=(x,y)
        # x goes along angle, y along radius
        assert res>1
        assert r1>0
        assert r2>r1
        start_angle = start_angle%360
        x = res+1
        deg_step = angle/(x-1)
        rad_step = np.deg2rad(deg_step)
        start_rad = np.deg2rad(start_angle)
        end_rad = start_rad + np.deg2rad(angle)
        corners = [
            (np.cos(start_rad)*r1, np.sin(start_rad)*r1),
            (np.cos(end_rad)*r1, np.sin(end_rad)*r1),
            (np.cos(start_rad)*r2, np.sin(start_rad)*r2),
            (np.cos(end_rad)*r2, np.sin(end_rad)*r2)
        ]
        lower_border = [(np.cos(start_rad + rad_step*i)*r1, np.sin(start_rad + rad_step*i)*r1) for i in range(x)] # -y
        upper_border = [(np.cos(start_rad + rad_step*i)*r2, np.sin(start_rad + rad_step*i)*r2) for i in range(x)] # -y
        
        # roughly square cells, growing linearly with radius
        r = r2-r1
        sizes = []
        d = r1
        y = 1
        width_scale = 2 * np.pi / x * (abs(angle)/360)
        while d<r2:
            width = d * width_scale
            sizes.append(width)
            d += width
            y += 1
        scale = (d-r1) / r
        sizes = [w/scale for w in sizes]
        # interpolation weights in [0,1]
        x_weights = [0] + [w/r for w in np.cumsum(sizes)]

        #print("square cells: (x=%d, y=%d),\ns=%s,\nw=%s"%(x,y,sizes,x_weights))

        #l_border = [() for i in range(res)]
        #print(lower_border)
        grid = generate_grid_vertices_2D([y,x],corners, [None, None, lower_border, upper_border], x_weights=x_weights, dtype=dtype)

        if offset=="CENTER":
            # via AABB
            vertex_corners = torch.tensor(corners, dtype=dtype)
            lower, _ = vertex_corners.min(dim=0)
            upper, _ = vertex_corners.max(dim=0)
            size = upper - lower
            center = lower + size*0.5
            offset = -center
            print(offset)
        
        if isinstance(offset, (list, tuple, np.ndarray)):
            offset = torch.tensor(offset, dtype=dtype)
        
        if isinstance(offset, torch.Tensor):
            assert offset.dtype==dtype
            assert offset.dim()==1
            assert offset.size(0)==2
            offset = offset.view(1,2,1,1)
            grid = grid + offset

        
        return grid

def make_cosX_grid(x, y, x_scale=1, y_scale=1, strength=1):
    
    baseline = np.asarray([[i*x_scale, 0] for i in range(x+1)])
    
    x_norm = (2*np.pi)/(y)
    offsets = np.asarray([((np.cos(x_norm*_)+1)*0.5*x_scale*strength, _*y_scale) for _ in range(y+1)])
    rows = [baseline + offsets[i] for i in range(y+1)]
    
    vertex_coords = np.asarray(rows) # HWC
    vertex_coords = np.moveaxis(vertex_coords, -1, 0) # CHW
    vertex_coords = np.expand_dims(vertex_coords, 0) #NCHW
    #vertex_coords = torch.ones((1,2,y,x), dtype=dtype, device=cuda_device) * vertex_coords #torch.tensor(vertex_coords, dtype=dtype, device=cuda_device)
    vertex_coords = torch.tensor(vertex_coords, dtype=dtype, device=cuda_device).contiguous()
    
    return vertex_coords

def grid_by_neighbor_linear(connected_grid,res_x,res_y,dtype,corner_lower, corner_upper,side):
    """
    This function is to generate the buffer grid for outlet in a linear way (so the physical length is not fixed). Firstly get the grid size of x-direction from connected_grid, then based on this size, generate the buffer grid. 
        - connected_grid: the grid that is connected to the buffer grid
        - res_x: the resolution of the buffer grid
        - res_y: noused but kept for later usage (connect to top grid)
        - base: the base of the exponential function
        - corner_lower: the lower corner of the buffer grid
        - corner_upper: the upper corner of the buffer grid
        - side: the side of the connected grid that we want to connect to
    """
    if side == "-x":
        prev_grid_phy = connected_grid[0, 0, 0, 1] - connected_grid[0, 0, 0, 0]
        refinement="END"
    if side == "+x":
        refinement="START"
        prev_grid_phy = connected_grid[0, 0, 0, -1] - connected_grid[0, 0, 0, -2]
    l_curr = res_x * prev_grid_phy
    x_weights=make_weights_linear(res_x,base=1,refinement=refinement)
    y_weight = get_weights_from_grid(connected_grid)[1]
    res_y=connected_grid.shape[2]
    corner_upper = [corner_lower[0]+l_curr, corner_upper[1]]
    corners = [tuple(corner_lower), (corner_upper[0], corner_lower[1]), (corner_lower[0], corner_upper[1]), tuple(corner_upper)]
    grid=generate_grid_vertices_2D(res=[res_y,res_x+1],dtype=dtype,corner_vertices=corners,x_weights=y_weight,y_weights=x_weights)
    #since res_y is from the grid, already stand for the number of vertices, so no need to plus 1
    return grid,l_curr

def grid_by_neighbor_exp(connected_grid,x,res_x,res_y,dtype,corner_lower, corner_upper,side):
    """
    This function is for generting the grid based neighbor grid to provide a smooth transition between grids and keeping the physical length by using exponential rate.
    Using the in_grow_factor to get a proper factor based on the neighbor grid size and the x(physical length) we want to get
    :connected_grid: the grid that we want to connect to
    :x: the physical length that we want to get
    :res_x: the resolution of the x direction
    :res_y: the resolution of the y direction (not nessary since the y direction should be the same with the connected grid, but for simplicity we still keep it)
    : corner_lower: the lower corner of the buffer grid
    : corner_upper: the upper corner of the buffer grid
    : side: the side of the connected grid that we want to connect to
    """
    if side == "-x":
        x0=connected_grid[0,0,0,1] - connected_grid[0,0,0,0]
        refinement="END"
    if side == "+x":
        x0=connected_grid[0,0,0,-1] - connected_grid[0,0,0,-2]
        refinement="START"
    in_grow_factor=growth_factor_from_sizes(res=res_x,x=x,x0=x0,print_info_fn=None)
    x_weight=make_weights_exp(res_x,base=in_grow_factor,refinement=refinement)
    y_weight=get_weights_from_grid(connected_grid)[1]
    res_y=connected_grid.shape[2]
    corners = [tuple(corner_lower), (corner_upper[0], corner_lower[1]), (corner_lower[0], corner_upper[1]), tuple(corner_upper)]
    #since res_y is from the grid, already stand for the number of vertices, so no need to plus 1
    grid=generate_grid_vertices_2D(res=[res_y,res_x+1],dtype=dtype,corner_vertices=corners,x_weights=y_weight,y_weights=x_weight)
    return grid,in_grow_factor

def get_weights_from_grid(grid):
    """This function is to get the weights tensor of the grid
    :grid: the grid tensor of shape [1, 2, res_y, res_x]
    output: the x_weights and y_weights of the grid in the shape of [res_y,res_x]
    ATTENTION: this res_x, and res_y are the number of vertices, not the number of cells. If you want to get the wights in 1D, just use x:x_weights[0,:], y: y_weights[:,0]
    """
    res_y, res_x = grid.shape[2], grid.shape[3]
    left_border = grid[0, :, :, 0].cpu()  
    right_border = grid[0, :, :, -1].cpu()  
    top_border = grid[0, :, 0, :].cpu()  
    bottom_border = grid[0, :, -1, :].cpu()  
    x_vals = grid[0, 0]  
    x_weights = (x_vals - left_border[0].unsqueeze(1)) / (right_border[0].unsqueeze(1) - left_border[0].unsqueeze(1))
    y_vals = grid[0, 1] 
    y_weights = (y_vals - top_border[1].unsqueeze(0)) / (bottom_border[1].unsqueeze(0) - top_border[1].unsqueeze(0) )
    return x_weights[0,:].tolist(), y_weights[:,0].tolist()

def growth_factor_from_sizes(res, x, x0, eps=1e-5, print_info_fn=None):
    from scipy.optimize import least_squares
    # res: number of cells
    # x: total size
    # x0: base cell size
    # finds exponent b s.t. x = sum[i=0, res-1](x0 * b**i) = x0*(1-b**res)/(1-b)
    if not (res>0 and x>x0 and x0>0 and eps>0): raise ValueError("invalid input")
    
    res = np.asarray([res])
    x = np.asarray([x])
    x0 = np.asarray([x0])
    
    if print_info_fn: print_info_fn("least squares: res=%.03e, x==%.03e, x0==%.03e, x/x0=%.03e.", res, x, x0, x/x0)
    
    if np.isclose(res, x/x0):
        return 1
    
    def func(b):
        return x0*(1-np.power(b,res))/(1-b) - x
    
    if res<x/x0:
        b0 = 1+eps
        lb = b0
        ub = 2
    else:
        b0 = 1-eps
        lb = 0+eps
        ub = b0
    
    b0 = np.asarray([b0])
    lb = np.asarray([lb])
    ub = np.asarray([ub])
    
    if print_info_fn: print_info_fn("least squares: b0=%.03e, lb==%.03e, ub==%.03e.", b0, lb, ub)
    
    b = least_squares(func, b0, bounds=(lb, ub))
    
    if print_info_fn: print_info_fn("least squares: %s.", b)
    
    return b.x

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
