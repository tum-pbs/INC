
from lib.util.profiling import SAMPLE
import torch

import PISOtorch
# import PISOtorch_diff
from solvers.solver_2d import PISOtorch_diff # differentiable wrapper for PISO

import numpy as np

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import lib.data.shapes as shapes
from lib.util.output import *
from lib.util.domain_io import save_domain

from lib.util.logging import get_logger
_LOG = get_logger("PISOsim")

__BACKEND = PISOtorch

# these must match the definition in 'PISO_multiblock_cuda.h'
NON_ORTHO_DIRECT_MATRIX = 1
NON_ORTHO_DIRECT_RHS = 2 # less stable than NON_ORTHO_DIRECT_MATRIX
NON_ORTHO_DIAGONAL_MATRIX = 4 # not implemented
NON_ORTHO_DIAGONAL_RHS = 8
NON_ORTHO_CENTER_MATRIX = 16

__NON_ORTHO_MODE = NON_ORTHO_CENTER_MATRIX | NON_ORTHO_DIRECT_MATRIX | NON_ORTHO_DIAGONAL_RHS # Bit flags
#__NON_ORTHO_MODE = 0 # orthogonal mode

__A_NON_ORTHO_MODE = __NON_ORTHO_MODE
__P_NON_ORTHO_MODE = __NON_ORTHO_MODE

def set_backend(diff=False):
    global __BACKEND
    if diff:
        __BACKEND = PISOtorch_diff
    else:
        __BACKEND = PISOtorch

def get_backend():
    return __BACKEND

def tensor_as_np(tensor):
    return tensor.detach().cpu().numpy()

def get_max_time_step(domain, time_step_target, CFL_cond=0.8, with_transformations=True):
    max_vel = domain.getMaxVelocity(True, with_transformations).cpu().numpy()
    max_time_step = CFL_cond/max_vel
    if max_time_step>=time_step_target:
        ss = 1
        ts = time_step_target
    else:
        ss = int(np.ceil(time_step_target / max_time_step))
        ts = time_step_target / ss
    return ts, ss


def getVelocityResultMaxMag(domain:PISOtorch.Domain):
    vel = domain.velocityResult
    vel = torch.reshape(vel, (domain.getSpatialDims(), domain.getTotalSize()))
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    if vel_max>0:
        return torch.max(torch.linalg.vector_norm(vel, dim=0)).cpu().numpy().tolist()
    return 0

def getVelocityResultMaxVel(domain:PISOtorch.Domain):
    vel = domain.velocityResult
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    return vel_max



def linear_solve_scipy(csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x=None):
    with SAMPLE("scipy linear solve"):
        A = csr_matrix((csrMat.value.cpu(), csrMat.index.cpu(), csrMat.row.cpu()), shape=[csrMat.getRows()]*2)
        with SAMPLE("linear solve"):
            x = spsolve(A, torch.flatten(rhs.cpu()))
        return torch.reshape(torch.tensor(x, dtype=rhs.dtype), rhs.size()).cuda()

def linear_solve_GPU(csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x:torch.Tensor=None, use_BiCG=False, tol=None, max_iter=5000,
                     matrix_rank_deficient=False, residual_reset_step=0, return_best_result=False):
    if __BACKEND==PISOtorch:
        convergence_criterion = PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED # RMSE of residual vector
        transpose = False # used for backprop
        print_residual = False # Print final solver residual information. Use for debugging. Only for CG.
        #return_best_result = False # saves best intermediate result and returns it instead of final result after max_iter if the solve does not converge. Only for CG.
        maxit_torch = torch.IntTensor([max_iter])
        if tol is None:
            if rhs.dtype==torch.float64:
                tol = 1e-8
            elif rhs.dtype==torch.float32:
                tol = 1e-5
        tol_torch = torch.ones([1], dtype = rhs.dtype)*tol
        with SAMPLE("GPU linear solve"):
            x_given = x is not None
            if not x_given:
                x = torch.zeros_like(rhs)
            if not rhs.eq(0).all():
                # SolveLinear(A:PISOtorch.CSRmatrix, RHS:torch.tensor, result:torch.tensor, max iterations, tolerance, convergence cirterion:PISOtorch.ConvergenceCriterion,
                #   use_BiCG, matrix_rank_deficient(do not use!), residual_reset_step(do not use!), transposeA, print residual information)
                it = PISOtorch.SolveLinear(csrMat, rhs, x, maxit_torch, tol_torch, convergence_criterion,
                        use_BiCG, matrix_rank_deficient, residual_reset_step, transpose,
                        print_residual, return_best_result)
            else:
                it = 0
                if x_given:
                    x.zero_()
            #_LOG.info("GPU solve took %d iterations.", it)
            if(it>=max_iter):
                #_LOG.warning("GPU solve did not converge after %d iterations.", it)
                _LOG.warning("GPU %sCG solve did not converge after %d iterations.", "Bi" if use_BiCG else "", it)
            return x, ((0<=it and it<max_iter) or return_best_result)
    else:
        if x is not None:
            _LOG.warning("x is ignored when using PISOtorch_diff.")
        return __BACKEND.linear_solve_GPU(csrMat, rhs, transpose=False, use_BiCG=use_BiCG, tol=tol, max_iter=max_iter, return_best_result=return_best_result), True

def linear_solve(csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x:torch.Tensor=None, use_BiCG=False, tol=None, max_iter=1000,
                 matrix_rank_deficient=False, residual_reset_step=0, use_scipy=False, return_best_result=False):
    if use_scipy:
        return linear_solve_scipy(csrMat, rhs, x), True
    else:
        return linear_solve_GPU(csrMat, rhs, x, use_BiCG, tol, max_iter, matrix_rank_deficient, residual_reset_step, return_best_result)

def get_varying_boundary_flux(bound, bound_idx):
    assert isinstance(bound, PISOtorch.VaryingDirichletBoundary)
    dims = bound.getSpatialDims()
    bound_axis = bound_idx//2
    if bound.hasTransform:
        shape = bound.getSizes() #x,y,z
        transform = bound.transform # NDHWC
        if True:
            inv_row_start = dims*dims + bound_axis*dims
            inv_row_end = inv_row_start + dims
            t_inv_row = transform[...,inv_row_start:inv_row_end].view(-1,1,dims) # (NDHW)1C
            J = transform[...,-1].view(-1, 1, 1) # (NDHW)11
            bound_vel = torch.moveaxis(bound.boundaryVelocity, 1, -1).view(-1,dims,1) # NCDHW -> NDHWC -> (NDHW)C1
            flux = J * torch.bmm(t_inv_row, bound_vel)
            
            return torch.sum(flux)
        else:
            t_inv = transform[...,dims*dims:dims*dims*2].view(-1,dims,dims)
            J = transform[...,-1].view(-1,1,1) # NDHWC -> (NDHW)11
            bound_vel = torch.moveaxis(bound.boundaryVelocity, 1, -1).view(-1,dims,1) # NCDHW -> NDHWC -> (NDHW)C1
            fluxes = torch.matmul(t_inv, bound_vel) * J
            flux = torch.reshape(fluxes, [1] + [shape[dim] for dim in range(dims-1, -1, -1)] + [dims]) #NDHWC
            return torch.sum(flux[...,bound_axis])
    else:
        return torch.sum(bound.boundaryVelocity[:,bound_axis])

# this breaks the outflow
def restrict_inflow(bound_vel, bound_idx):
    # clamp boundary flux s.t. there is no inflow
    bound_axis = bound_idx>>1
    bound_dir = bound_idx&1
    # only for orthogonal transforms
    dims = bound_vel.dim()-2
    channels = list(bound_vel.split(dims, dim=1))
    flux = channels[bound_axis]
    zero = torch.tensor(0, dtype=bound_vel.dtype, device=bound_vel.device)
    if bound_dir==0: # lower bound, inlfow is positive
        flux = torch.mimimum(flux, zero)
    else:
        flux = torch.maximum(flux, zero)
    channels[bound_axis] = flux
    return torch.cat(channels, dim=1)



# see also: https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2022/LeandroLucchese/Report_Lucchese.pdf
def update_advective_boundaries(domain, bounds, velms, dt):
    #_LOG.info("adective bound update: %d boundaries", len(bounds))
    with torch.no_grad():
        #if any([block.hasTransform for block in domain.getBlocks()]):
        #    raise RuntimeError("update_advective_boundaries does not support transformations.")
        fixed_boundary_flux = torch.zeros([1], dtype=torch.float32, device=cuda_device)
        variable_boundary_flux = torch.zeros([1], dtype=torch.float32, device=cuda_device)

        dims = domain.getSpatialDims()
        boundaries = []
        for blockIdx in range(domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            for boundIdx in range(dims*2):
                bound = block.getBoundary(boundIdx)
                if isinstance(bound, (PISOtorch.VaryingDirichletBoundary, PISOtorch.StaticDirichletBoundary)):
                    boundaries.append((block, boundIdx, bound))
                if isinstance(bound, PISOtorch.FixedBoundary):
                    raise TypeError("FixedBoundary is not supported in advective outflow update")
        
        fixed_boundaries = [_ for _ in boundaries if _[2] not in bounds]
        variable_boundaries = [_ for _ in boundaries if _[2] in bounds]
        
        #_LOG.info("%d fixed, %d variable", len(fixed_boundaries), len(variable_boundaries))

        for block, boundIdx, bound in fixed_boundaries:
            if isinstance(bound, PISOtorch.VaryingDirichletBoundary):
            # TODO: non-ortho?
                #boundary_flux = torch.sum(bound.boundaryVelocity[:,boundIdx//2])
                boundary_flux = get_varying_boundary_flux(bound, boundIdx)
            elif isinstance(bound, PISOtorch.StaticDirichletBoundary):
                boundary_size = block.getSizes()
                boundary_cells = 1
                for dim in range(dims):
                    if not dim== boundIdx//2:
                        boundary_cells *= boundary_size[dim]
                boundary_flux = bound.boundaryVelocity[boundIdx//2] * boundary_cells
                #boundary_flux = 0
            if boundIdx%2==0:
                fixed_boundary_flux -= boundary_flux
            else:
                fixed_boundary_flux += boundary_flux

            
        for block, boundIdx, bound in variable_boundaries:
            if isinstance(bound, PISOtorch.VaryingDirichletBoundary):
                if boundIdx==0:
                    vel_slice = block.velocity[...,:1]
                    scal_slice = block.passiveScalar[...,:1]
                elif boundIdx==1:
                    vel_slice = block.velocity[...,-1:]
                    scal_slice = block.passiveScalar[...,-1:]
                elif boundIdx==2:
                    vel_slice = block.velocity[...,:1,:]
                    scal_slice = block.passiveScalar[...,:1,:]
                elif boundIdx==3:
                    vel_slice = block.velocity[...,-1:,:]
                    scal_slice = block.passiveScalar[...,-1:,:]
                elif boundIdx==4:
                    vel_slice = block.velocity[...,:1,:,:]
                    scal_slice = block.passiveScalar[...,:1,:,:]
                elif boundIdx==5:
                    vel_slice = block.velocity[...,-1:,:,:]
                    scal_slice = block.passiveScalar[...,-1:,:,:]
                else:
                    raise RuntimeError
                vel_m = torch.abs(velms[bounds.index(bound)])
                alpha = dt*2*vel_m # dt * flow speed / distance(center, face)
                if bound.hasTransform:
                    bound_axis = (boundIdx>>1)
                    matrix_element = dims*dims + bound_axis*dims + bound_axis
                    alpha = alpha * bound.transform[...,matrix_element] # inverse cell size in boundary-normal direction, assumes orthogonal transformation
                t = 1 - 1/(1+alpha) # interpolation weight
                #_LOG.info("advective bound interpolation weight: %s", t)
                vel_bound = bound.boundaryVelocity
                vel_bound_update = vel_bound - t*(vel_bound - vel_slice)
                vel_bound.copy_(vel_bound_update)
                scal_bound = bound.boundaryScalar
                scal_bound.copy_(scal_bound - t*(scal_bound - scal_slice))
                #boundary_flux = torch.sum(vel_bound[:,boundIdx//2])
                boundary_flux = get_varying_boundary_flux(bound, boundIdx)
                #_LOG.info("block %s var bound %d flux: %s", block.name, boundIdx, boundary_flux)
                if boundIdx%2==0:
                    variable_boundary_flux -= boundary_flux
                else:
                    variable_boundary_flux += boundary_flux
                #_LOG.info("VDB after update: flux %s", boundary_flux)
            else:
                raise TypeError
        
        scale_all = False
        #compensation_additive = False
        #_LOG.info("Fluxes: fixed %s, var %s", fixed_boundary_flux, variable_boundary_flux)
        if not torch.allclose(fixed_boundary_flux+variable_boundary_flux, torch.zeros_like(fixed_boundary_flux)):
            flux_scale = -fixed_boundary_flux/variable_boundary_flux
            d_flux = fixed_boundary_flux - variable_boundary_flux
            #_LOG.info("FluxScale: %s", flux_scale)

            for block, boundIdx, bound in variable_boundaries:
                # TODO: scale everyting or only the boundary normal component?
                if scale_all:
                    bound.boundaryVelocity.copy_(bound.boundaryVelocity*flux_scale)
                else:
                    # TODO: non-ortho transform
                    vel_comps = list(torch.split(bound.boundaryVelocity, domain.getSpatialDims(), dim=1))
                    vel_comps[boundIdx//2] = vel_comps[boundIdx//2]*flux_scale
                    bound.boundaryVelocity.copy_(torch.cat(vel_comps, axis=1))
                #_LOG.info("VDB after update: flux %s\n%s", torch.sum(bound.boundaryVelocity[:,boundIdx//2]), bound.boundaryVelocity)
                #_LOG.info("VDB after compensation: flux %s", torch.sum(bound.boundaryVelocity[:,boundIdx//2]))

def update_advective_boundaries_static(domain, bounds, velms, dt):

    boundaries = []
    for blockIdx in range(domain.getNumBlocks()):
        block = domain.getBlock(blockIdx)
        for boundIdx in range(domain.getSpatialDims()*2):
            bound = block.getBoundary(boundIdx)
            if isinstance(bound, (PISOtorch.VaryingDirichletBoundary, PISOtorch.StaticDirichletBoundary)):
                boundaries.append((boundIdx, bound))
    
    variable_boundaries = [_ for _ in boundaries if _[1] in bounds]

        
    for boundIdx, bound in variable_boundaries:
        if isinstance(bound, PISOtorch.VaryingDirichletBoundary):
            if boundIdx==0:
                scal_slice = block.passiveScalar[...,:1]
            elif boundIdx==1:
                scal_slice = block.passiveScalar[...,-1:]
            elif boundIdx==2:
                scal_slice = block.passiveScalar[...,:1,:]
            elif boundIdx==3:
                scal_slice = block.passiveScalar[...,-1:,:]
            elif boundIdx==4:
                scal_slice = block.passiveScalar[...,:1,:,:]
            elif boundIdx==5:
                scal_slice = block.passiveScalar[...,-1:,:,:]
            else:
                raise RuntimeError
            vel_m = torch.abs(velms[bounds.index(bound)])
            scal_bound = bound.boundaryScalar
            scal_bound.copy_(scal_bound - (dt*2*vel_m)*(scal_bound - scal_slice))
        else:
            raise TypeError

def advect_static(domain:PISOtorch.Domain, *, steps:int=1, advect_non_ortho_steps:int=1, time_step:float=1.0, scipy_solve_advection=False, prep_fn=None, STOP_FN=lambda: False):
    
    #prev_as_initial_guess_advection = False
    solve_ok = True

    advect_non_ortho_reuse_result = True
    
    if isinstance(steps, torch.Tensor):
        steps = steps.numpy()[0]
    
    for step in range(steps):
        with SAMPLE("advect static"):
            if prep_fn: prep_fn(domain=domain, local_step=step, time_step=time_step)
            
            domain.UpdateDomainData()
            
            #_LOG.info("SetupMatrix...")
            __BACKEND.SetupAdvectionMatrix(domain, time_step, __A_NON_ORTHO_MODE)
            #_LOG.info("Done")
            __BACKEND.CopyScalarResultFromBlocks(domain) # needed for non-ortho components on RHS
            
            last_scalar_result = 0
            for no_step in range(advect_non_ortho_steps):
                #_LOG.info("SetupAdvectionScalar...")
                __BACKEND.SetupAdvectionScalar(domain, time_step, __A_NON_ORTHO_MODE)
                #_LOG.info("Done")
                
                x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.scalarResult
                #_LOG.info("linear_solve...")
                scalarResult, solve_ok = linear_solve(domain.C, domain.scalarRHS, x=x, use_BiCG=True, use_scipy=scipy_solve_advection)
                #_LOG.info("Done")
                del x
                
                domain.setScalarResult(scalarResult)
                domain.UpdateDomainData()
                
                if False: #DEBUG
                    dif_p = torch.abs(domain.scalarResult.detach() - last_scalar_result)
                    _LOG.info("s-no-step %d diff: mean=%.03e, max=%.03e", no_step, torch.mean(dif_p).cpu().numpy(), torch.max(dif_p).cpu().numpy())
                    last_scalar_result = domain.scalarResult.detach().clone()
                
                if not solve_ok or STOP_FN():
                    return solve_ok
            
            __BACKEND.CopyScalarResultToBlocks(domain) #set final result to blocks for next iteration/step
            
            if not solve_ok or STOP_FN():
                return solve_ok
                
            del last_scalar_result
            #del base_scalar_rhs
    
    return solve_ok

def advect_velocity(domain:PISOtorch.Domain, *, steps:int=1, time_step:float=1.0, advect_non_ortho_steps:int=1, scipy_solve_advection=False, prep_fn=None, STOP_FN=lambda: False):
    
    #prev_as_initial_guess_advection = False
    solve_ok = True
    advect_non_ortho_reuse_result = True
    
    if isinstance(steps, torch.Tensor):
        steps = steps.numpy()[0]
    
    for step in range(steps):
        with SAMPLE("advect velocity"):
            if prep_fn: prep_fn(domain=domain, local_step=step, time_step=time_step)
            
            apply_pressure_gradient = False
            
            domain.UpdateDomainData()
            
            __BACKEND.SetupAdvectionMatrix(domain, time_step, __A_NON_ORTHO_MODE)
            __BACKEND.CopyVelocityResultFromBlocks(domain) # needed for non-ortho components on RHS
            
            
            last_velocity_result = 0
            for no_step in range(advect_non_ortho_steps):
                __BACKEND.SetupAdvectionVelocity(domain, time_step, __A_NON_ORTHO_MODE, apply_pressure_gradient)
                
                x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.velocityResult
                velocityResult, solve_ok = linear_solve(domain.C, domain.velocityRHS, x=x, use_BiCG=True, use_scipy=scipy_solve_advection)
                del x
                
                domain.setVelocityResult(velocityResult)
                domain.UpdateDomainData()
                
                if False: #DEBUG
                    dif_p = torch.abs(domain.velocityResult.detach() - last_velocity_result)
                    _LOG.info("v-no-step %d diff: mean=%.03e, max=%.03e", no_step, torch.mean(dif_p).cpu().numpy(), torch.max(dif_p).cpu().numpy())
                    last_velocity_result = domain.velocityResult.detach().clone()
                
                if not solve_ok or STOP_FN():
                    return solve_ok
            
            __BACKEND.CopyVelocityResultToBlocks(domain)
            
            #domain.UpdateDomainData()
            
            if not solve_ok or STOP_FN():
                return solve_ok
            
            del last_scalar_result
    
    return solve_ok

def PISO_div_free(domain:PISOtorch.Domain, *, steps:int=1, time_step:float=1.0,
        pressure_non_ortho_steps:int=1, pressure_return_best_result:bool=False,
        scipy_solve_pressure=False, pressure_use_BiCG=False,
        velocity_corrector=0,
        prep_fn=None, STOP_FN=lambda: False):
    
    #prev_as_initial_guess_pressure = False
    corrector_steps = 1
    vcv = velocity_corrector

    if isinstance(steps, torch.Tensor):
        steps = steps.numpy()[0]
    if isinstance(corrector_steps, torch.Tensor):
        corrector_steps = corrector_steps.numpy()[0]
    
    for step in range(steps):
        with SAMPLE("PISO step"):
            if prep_fn: prep_fn(domain=domain, local_step=step, time_step=time_step)
            
            __BACKEND.CopyVelocityResultFromBlocks(domain)
            for blockIdx in range(0, domain.getNumBlocks()):
                domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
                domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
            domain.UpdateDomainData()
            # set up with u=0
            __BACKEND.SetupAdvectionMatrix(domain, time_step, __A_NON_ORTHO_MODE)
            # for debug output
            #PISOtorch.SetupAdvectionScalar(domain, time_step)
            #PISOtorch.SetupAdvectionVelocity(domain, time_step)
            
            __BACKEND.CopyVelocityResultToBlocks(domain)
            # u^0=u*

            for cstep in range(corrector_steps):
                with SAMPLE("corrector step"):
                    #__BACKEND.SetupPressureCorrection(domain, time_step, __P_NON_ORTHO_MODE)
                    __BACKEND.SetupPressureMatrix(domain, time_step, __P_NON_ORTHO_MODE)
                    
                    last_pressure_result = 0
                    for pstep in range(pressure_non_ortho_steps):
                        if pstep==0:
                            __BACKEND.SetupPressureRHS(domain, time_step, __P_NON_ORTHO_MODE) # build rhs (vector field) and div(rhs) + non-ortho
                        else:
                            __BACKEND.SetupPressureRHSdiv(domain, time_step, __P_NON_ORTHO_MODE) # build only div(rhs) + non-ortho from existing rhs vector field
                        
                        x = None if (pstep==0 or not pressure_reuse_result) else domain.pressureResult
                        pressureResult, solve_ok = linear_solve(domain.P, domain.pressureRHSdiv, x=x,
                            use_BiCG=pressure_use_BiCG, use_scipy=scipy_solve_pressure, tol=pressure_tol, return_best_result=pressure_return_best_result) #, x=domain.pressureResult
                        del x
                        
                        domain.setPressureResult(pressureResult)
                        domain.UpdateDomainData()
                        #DEBUG
                        #solve_ok = True
                        __BACKEND.CopyPressureResultToBlocks(domain) # currently the non-ortho coefficients in pressure RHS still use block.pressure, not domain.pressureResult.
                        
                        if not solve_ok:
                            return solve_ok
                        
                        if STOP_FN():
                            break
                    
                    del last_pressure_result
                    
                    __BACKEND.CorrectVelocity(domain, time_step, version=vcv)
            
            __BACKEND.CopyVelocityResultToBlocks(domain)
            
            if not solve_ok or STOP_FN():
                return solve_ok
    
    return solve_ok

def PISO_div_free_v2(domain:PISOtorch.Domain, *, steps:int=1, time_step:float=1.0,
        pressure_non_ortho_steps:int=1, pressure_return_best_result:bool=False,
        scipy_solve_pressure=False, pressure_use_BiCG=False, pressure_tol=None,
        velocity_corrector=0,
        prep_fn=None, STOP_FN=lambda: False):
    
    #prev_as_initial_guess_pressure = False
    corrector_steps = 1
    pressure_use_face_transform = False
    vcv = velocity_corrector
    
    # overwrite time step
    time_step = torch.tensor([1], device=cpu_device, dtype=domain.A.dtype)

    if isinstance(steps, torch.Tensor):
        steps = steps.numpy()[0]
    if isinstance(corrector_steps, torch.Tensor):
        corrector_steps = corrector_steps.numpy()[0]
    
    for step in range(steps):
        with SAMPLE("PISO step"):
            _run_prep_fn(prep_fn, "PRE", domain=domain, local_step=step, time_step=time_step)
            
            domain.A.copy_(torch.ones_like(domain.A))
            __BACKEND.CopyVelocityResultFromBlocks(domain)

            for cstep in range(corrector_steps):
                with SAMPLE("corrector step"):
                    domain.pressureRHS.copy_(domain.velocityResult)
                    __BACKEND.SetupPressureMatrix(domain, time_step, __P_NON_ORTHO_MODE, pressure_use_face_transform)
                    
                    last_pressure_result = 0
                    for pstep in range(pressure_non_ortho_steps):
                        # build only div(rhs) + non-ortho from existing rhs vector field
                        __BACKEND.SetupPressureRHSdiv(domain, time_step, __P_NON_ORTHO_MODE, pressure_use_face_transform) 
                        if not torch.all(torch.isfinite(domain.P.value)): _LOG.warning("P.value is not finite.")
                        if not torch.all(torch.isfinite(domain.pressureRHSdiv)): #_LOG.warning("pressureRHSdiv is not finite.")
                            not_finite = torch.logical_not(torch.isfinite(domain.pressureRHSdiv))
                        #    #_LOG.warning("scalarRHS is not finite: %s/%s are not finite.\n%s", torch.sum(not_finite.int()), domain.getTotalSize(), not_finite.int().nonzero())
                            _LOG.warning("pressureRHSdiv is not finite: %s/%s are not finite.", torch.sum(not_finite.int()), domain.getTotalSize())
                        #    return False
                        
                        x = None if (pstep==0 or not pressure_reuse_result) else domain.pressureResult
                        pressureResult, solve_ok = linear_solve(domain.P, domain.pressureRHSdiv, x=x,
                            use_BiCG=pressure_use_BiCG, use_scipy=scipy_solve_pressure, tol=pressure_tol, return_best_result=pressure_return_best_result) #, x=domain.pressureResult
                        del x
                        
                        if not torch.all(torch.isfinite(pressureResult)):
                            not_finite = torch.logical_not(torch.isfinite(pressureResult))
                            _LOG.warning("pressureResult is not finite: %s/%s are not finite.", torch.sum(not_finite.int()), domain.getTotalSize())
                        
                        domain.setPressureResult(pressureResult)
                        domain.UpdateDomainData()
                        #DEBUG
                        #solve_ok = True
                        
                        if not solve_ok:
                            return solve_ok
                        
                        if STOP_FN():
                            break
                    
                    del last_pressure_result
                    
                    __BACKEND.CopyPressureResultToBlocks(domain)
                    
                    __BACKEND.CorrectVelocity(domain, time_step, version=vcv)
            
            __BACKEND.CopyVelocityResultToBlocks(domain)
            
            if not solve_ok or STOP_FN():
                return solve_ok
    
    return solve_ok


def append_prep_fn(prep_fn, name, fn):
    assert isinstance(prep_fn, dict)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])
        
        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [prep_fn[name], fn]
        else:
            prep_fn[name].append(fn)
    return prep_fn

def prepend_prep_fn(prep_fn, name, fn):
    assert isinstance(prep_fn, dict)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])
        
        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [fn, prep_fn[name]]
        else:
            prep_fn[name].insert(0, fn)
    return prep_fn

def _run_prep_fn(prep_fn, name, **kwargs):
    if (prep_fn is not None) and (name in prep_fn) and (prep_fn[name] is not None):
        fns = prep_fn[name]
        if not isinstance(fns, (list, tuple)):
            fns = [fns]
        for fn in fns:
            if fn is not None: fn(**kwargs)


class SimInfo:
    def __init__(self):
        self.total_step = 0
    def end_sim_step(self):
        self.total_step +=1

def PISO_split_step(domain:PISOtorch.Domain, *, steps:int=1, corrector_steps:int=2, time_step:float=1.0, density_viscosity=None,
        advect_non_ortho_steps:int=1, pressure_non_ortho_steps:int=1, pressure_return_best_result:int=False,
        scipy_solve_advection=False, scipy_solve_pressure=False, pressure_use_BiCG=False, advection_tol=None, pressure_tol=None,
        velocity_corrector=1,
        prep_fn=None, convergence_tol=None, STOP_FN=lambda: False, sim_info:SimInfo=None):
    
    #prev_as_initial_guess_advection = False
    #prev_as_initial_guess_pressure = False
    solve_ok = True
    assert steps>0
    #assert corrector_steps>0
    assert time_step>0
    assert advect_non_ortho_steps>0
    assert pressure_non_ortho_steps>0
    advect_non_ortho_reuse_result = True
    pressure_reuse_result = True # speeds up pressure_non_ortho_steps, no difference in result noticed.
    pressure_use_face_transform = False
    vcv = velocity_corrector # velocity corrector version. use different pressure gradients: 0 default (finite volume), 1 for finite differences, 4 for correcting fluxes (orthogonal), 5 for finite volume, 6 FVM with face transformations
    pressure_dp = False
    pressure_time_step_normalized = False # wether the pressure field should be independent of the current time step. False is more stable.
    
    # DEBUG
    diffusion_only = False
    if diffusion_only:
        _LOG.info("Advection disabled!")

    if not isinstance(prep_fn, dict):
        prep_fn = {"PRE": prep_fn} if prep_fn is not None else {}
    
    if isinstance(steps, torch.Tensor):
        steps = steps.numpy()[0]
    if isinstance(corrector_steps, torch.Tensor):
        corrector_steps = corrector_steps.numpy()[0]
    
    for step in range(steps):
        with SAMPLE("PISO step"):
            #_LOG.info("Substep %d", step)
            if convergence_tol is not None:
                __BACKEND.CopyVelocityResultFromBlocks(domain)
                last_vel = domain.velocityResult.clone().detach()
            
            #if "PRE" in prep_fn: prep_fn["PRE"](domain=domain, local_step=step, time_step=time_step)
            _run_prep_fn(prep_fn, "PRE", domain=domain, local_step=step, time_step=time_step, sim_info=sim_info)
            #_LOG.info("UpdateDomainData")
            domain.UpdateDomainData()
            
            with SAMPLE("Advect scalar"):
                if density_viscosity is not None:
                    viscosity = domain.viscosity
                    domain.setViscosity(density_viscosity)
                
                #_LOG.info("SetupAdvectionMatrix")
                if diffusion_only:
                    __BACKEND.CopyVelocityResultFromBlocks(domain)
                    for block in domain.getBlocks():
                        block.velocity.zero_()
                __BACKEND.SetupAdvectionMatrix(domain, time_step, __A_NON_ORTHO_MODE)
                if diffusion_only:
                    __BACKEND.CopyVelocityResultToBlocks(domain)
                #_LOG.info("CopyScalarResultFromBlocks")
                
                if(__NON_ORTHO_MODE==0): # orthogonal version with gradient/backprop support
                    
                    __BACKEND.SetupAdvectionScalar(domain, time_step, __A_NON_ORTHO_MODE)
                    
                    _run_prep_fn(prep_fn, "POST_SCALAR_SETUP", domain=domain, local_step=step, time_step=time_step)
                    
                    scalarResult, solve_ok = linear_solve(domain.C, domain.scalarRHS, x=None, use_BiCG=True, use_scipy=scipy_solve_advection)
                    
                    domain.setScalarResult(scalarResult)
                    domain.UpdateDomainData()
                    
                else:
                    __BACKEND.CopyScalarResultFromBlocks(domain) # needed for non-ortho components on RHS
                    
                    last_scalar_result = 0
                    for no_step in range(advect_non_ortho_steps):
                        #_LOG.info("SetupAdvectionScalar")
                        __BACKEND.SetupAdvectionScalar(domain, time_step, __A_NON_ORTHO_MODE)
                        #if not torch.all(torch.isfinite(domain.C.value)): _LOG.warning("C.value is not finite.")
                        #if not torch.all(torch.isfinite(domain.scalarRHS)):
                        #    not_finite = torch.logical_not(torch.isfinite(domain.scalarRHS))
                        #    #_LOG.warning("scalarRHS is not finite: %s/%s are not finite.\n%s", torch.sum(not_finite.int()), domain.getTotalSize(), not_finite.int().nonzero())
                        #    _LOG.warning("scalarRHS is not finite: %s/%s are not finite.", torch.sum(not_finite.int()), domain.getTotalSize())
                        #    return False
                        
                        _run_prep_fn(prep_fn, "POST_SCALAR_SETUP", domain=domain, local_step=step, time_step=time_step)
                        
                        x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.scalarResult
                        #_LOG.info("linear_solve")
                        scalarResult, solve_ok = linear_solve(domain.C, domain.scalarRHS, x=x, use_BiCG=True, use_scipy=scipy_solve_advection)
                        del x
                        
                        domain.setScalarResult(scalarResult)
                        domain.UpdateDomainData()
                        
                        if False: #DEBUG
                            dif_p = torch.abs(domain.scalarResult.detach() - last_scalar_result)
                            #_LOG.info("s-no-step %d diff: mean=%.03e, max=%.03e", no_step, torch.mean(dif_p).cpu().numpy(), torch.max(dif_p).cpu().numpy())
                            last_scalar_result = domain.scalarResult.detach().clone()
                        
                        if not solve_ok or STOP_FN():
                            return solve_ok
                    
                    del last_scalar_result
                
                #_LOG.info("CopyScalarResultToBlocks")
                __BACKEND.CopyScalarResultToBlocks(domain)
            

            with SAMPLE("Advect velocity"):
                
                # DON'T use pressure from previous corrector steps, otherwise it's applied twice
                apply_pressure_gradient = False
                #for block in domain.getBlocks():
                #    block.pressure.zero_()
                
                if density_viscosity is not None:
                    domain.setViscosity(viscosity)
                    __BACKEND.SetupAdvectionMatrix(domain, time_step, __A_NON_ORTHO_MODE)
                
                if(__NON_ORTHO_MODE==0): # orthogonal version with gradient/backprop support
                    __BACKEND.SetupAdvectionVelocity(domain, time_step, 0, apply_pressure_gradient)

                    _run_prep_fn(prep_fn, "POST_VELOCITY_SETUP", domain=domain, local_step=step, time_step=time_step)
                    
                    velocityResult, solve_ok = linear_solve(domain.C, domain.velocityRHS, x=None, use_BiCG=True, use_scipy=scipy_solve_advection)
                    
                    domain.setVelocityResult(velocityResult)
                    domain.UpdateDomainData()
                    
                else:
                    #_LOG.info("CopyVelocityResultFromBlocks")
                    __BACKEND.CopyVelocityResultFromBlocks(domain) # needed for non-ortho components on RHS
                    
                    last_velocity_result = 0
                    for no_step in range(advect_non_ortho_steps):
                        #_LOG.info("SetupAdvectionVelocity")
                        __BACKEND.SetupAdvectionVelocity(domain, time_step, __A_NON_ORTHO_MODE, apply_pressure_gradient)

                        _run_prep_fn(prep_fn, "POST_VELOCITY_SETUP", domain=domain, local_step=step, time_step=time_step)
                        
                        x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.velocityResult
                        #_LOG.info("linear_solve")
                        velocityResult, solve_ok = linear_solve(domain.C, domain.velocityRHS, x=x, use_BiCG=True, use_scipy=scipy_solve_advection)
                        del x
                        
                        domain.setVelocityResult(velocityResult)
                        domain.UpdateDomainData()
                        
                        if False: #DEBUG
                            dif_p = torch.abs(domain.velocityResult.detach() - last_velocity_result)
                            #_LOG.info("v-no-step %d diff: mean=%.03e, max=%.03e", no_step, torch.mean(dif_p).cpu().numpy(), torch.max(dif_p).cpu().numpy())
                            last_velocity_result = domain.velocityResult.detach().clone()
                        
                        if not solve_ok or STOP_FN():
                            return solve_ok
                        
                    del last_velocity_result
                    
                
                #CopyVelocityResultToBlocks(domain) not yet, original vel still needed for pressure rhs
                
                if not solve_ok:
                    return solve_ok

            _run_prep_fn(prep_fn, "POST_PREDICTION", domain=domain, local_step=step, time_step=time_step)
            
            for cstep in range(corrector_steps):
                with SAMPLE("corrector step"):
                    
                    if(__NON_ORTHO_MODE==0): # orthogonal version with gradient/backprop support
                        __BACKEND.SetupPressureCorrection(domain, time_step, 0, pressure_use_face_transform, timeStepNorm=pressure_time_step_normalized)
                        
                        _run_prep_fn(prep_fn, "POST_PRESSURE_SETUP", domain=domain, local_step=step, time_step=time_step)
                        
                        pressureResult, solve_ok = linear_solve(domain.P, domain.pressureRHSdiv, x=None, matrix_rank_deficient=False, residual_reset_step=0,
                            use_BiCG=pressure_use_BiCG, use_scipy=scipy_solve_pressure, tol=pressure_tol, return_best_result=pressure_return_best_result)
                        
                        if not solve_ok:
                            return solve_ok
                        
                        domain.setPressureResult(pressureResult)
                        domain.UpdateDomainData()
                        
                        _run_prep_fn(prep_fn, "POST_PRESSURE_RESULT", domain=domain, local_step=step, time_step=time_step)
                        
                    else: # non-ortho version
                        #_LOG.info("SetupPressureMatrix")
                        __BACKEND.SetupPressureMatrix(domain, time_step, __P_NON_ORTHO_MODE, pressure_use_face_transform)
                        
                        last_pressure_result = 0
                        for pstep in range(pressure_non_ortho_steps):
                            #__BACKEND.SetupPressureCorrection(domain, time_step, __P_NON_ORTHO_MODE)
                            if pstep==0:
                                #_LOG.info("SetupPressureRHS")
                                # build rhs (vector field) and div(rhs) + non-ortho
                                __BACKEND.SetupPressureRHS(domain, time_step, __P_NON_ORTHO_MODE, pressure_use_face_transform, timeStepNorm=pressure_time_step_normalized)
                            else:
                                #_LOG.info("SetupPressureRHSdiv")
                                # build only div(rhs) + non-ortho from existing rhs vector field
                                __BACKEND.SetupPressureRHSdiv(domain, time_step, __P_NON_ORTHO_MODE, pressure_use_face_transform, timeStepNorm=pressure_time_step_normalized)

                            _run_prep_fn(prep_fn, "POST_PRESSURE_SETUP", domain=domain, local_step=step, time_step=time_step)

                            #_LOG.info("Start pressure solve #%d", cstep)
                            x = None if (pstep==0 or not pressure_reuse_result) else domain.pressureResult
                            if pressure_dp:
                                _LOG.info("Pressure solve is double precision.")
                                P = domain.P.toType(torch.float64)
                                pressureRHSdiv = domain.pressureRHSdiv.to(torch.float64)
                                if x is not None: x = x.to(torch.float64)
                                pressureResult, solve_ok = linear_solve(P, pressureRHSdiv, x=x,
                                    use_BiCG=pressure_use_BiCG, use_scipy=scipy_solve_pressure, tol=pressure_tol, return_best_result=pressure_return_best_result) #, x=domain.pressureResult
                                pressureResult = pressureResult.to(domain.pressureRHSdiv.dtype)
                                del P
                                del pressureRHSdiv
                            else:
                                #_LOG.info("linear_solve")
                                pressureResult, solve_ok = linear_solve(domain.P, domain.pressureRHSdiv, x=x, matrix_rank_deficient=False, residual_reset_step=100,
                                    use_BiCG=pressure_use_BiCG, use_scipy=scipy_solve_pressure, tol=pressure_tol, return_best_result=pressure_return_best_result) #, x=domain.pressureResult
                            del x
                            #solve_ok = True #DEBUG
                            
                            domain.setPressureResult(pressureResult)
                            domain.UpdateDomainData()
                            
                            if False: #DEBUG
                                dif_p = torch.abs(domain.pressureResult.detach() - last_pressure_result)
                                #_LOG.info("pstep %d diff: mean=%.03e, max=%.03e", pstep, torch.mean(dif_p).cpu().numpy(), torch.max(dif_p).cpu().numpy())
                                last_pressure_result = domain.pressureResult.detach().clone()

                            _run_prep_fn(prep_fn, "POST_PRESSURE_RESULT", domain=domain, local_step=step, time_step=time_step)
                            
                            if not solve_ok:
                                return solve_ok
                            
                            if STOP_FN():
                                break
                        
                        del last_pressure_result

                    _run_prep_fn(prep_fn, "POST_PRESSURE_NON_ORTHO", domain=domain, local_step=step, time_step=time_step)
                    
                    # if False:
                        # _LOG.info("DEBUG: using const 1 pressure.")
                        # domain.setPressureResult(torch.ones_like(domain.pressureResult))
                        # domain.UpdateDomainData()

                    __BACKEND.CopyPressureResultToBlocks(domain)

                    #_LOG.info("CorrectVelocity")
                    #if True:
                    __BACKEND.CorrectVelocity(domain, time_step, version=vcv, timeStepNorm=pressure_time_step_normalized) #vcv
                    # else:
                        # use_FVM = True
                        # gradient_interpolation = 3
                        # _LOG.info("DEBUG: using pressure gradient as velocity result: FVM %s, lerp %d", use_FVM, gradient_interpolation)
                        # p_grad = PISOtorch.ComputePressureGradient(domain, use_FVM, gradient_interpolation) # useFVM
                        # domain.setVelocityResult(p_grad)
                        # domain.UpdateDomainData()

                    _run_prep_fn(prep_fn, "POST_VELOCITY_CORRECTION", domain=domain, local_step=step, time_step=time_step)
                        
                    if STOP_FN():
                        break
            
            __BACKEND.CopyVelocityResultToBlocks(domain)

            _run_prep_fn(prep_fn, "POST", domain=domain, local_step=step, time_step=time_step)
            
            if convergence_tol is not None:
                step_max_diff = torch.max(torch.abs(last_vel - domain.velocityResult)).cpu().numpy()
                if step_max_diff<convergence_tol:
                    _LOG.info("Simulation step max difference is under convergence tolerance.")
                    solve_ok = False # to stop sim
                #elif (step+1)==steps: #debug
                #    ts = time_step.numpy()[0]
                #    _LOG.info("Simulation step max difference is %.02e. Normalized with time step %.02e: %.02e.", step_max_diff, ts, step_max_diff/ts)
            
            if not solve_ok or STOP_FN():
                break

            sim_info.end_sim_step()
    
    return solve_ok


def PISO_adaptive_step_v2(domain:PISOtorch.Domain, *, time_step:float=1.0, CFL_cond=0.8, max_subsetps=1000, STOP_FN=lambda: False, **split_step_kwargs):
    
    time_step_target = time_step
    substep = 0
    warned = False
    while time_step_target>0 and not np.isclose(time_step_target, 0):
        with SAMPLE("adaptive step"):
            max_vel = domain.getMaxVelocity(True, True)
            max_vel_np = max_vel.cpu().numpy()
            
            if np.isclose(max_vel_np, 0):
                max_time_step = time_step_target
            else:
                max_time_step = CFL_cond / max_vel_np
            
            if max_time_step>=time_step_target:
                substeps = 1
                ts = time_step_target
            else:
                substeps = int(np.ceil(time_step_target / max_time_step))
                ts = time_step_target / substeps
            
            time_step_target -= ts
            ts = ts *torch.ones([1], dtype=domain.getBlock(0).velocity.dtype, device=cpu_device)
            
            #_LOG.info("Adaptive step v2: maxVel %f, substep %d, timestep %f, remaining time %f", max_vel_np, substep, ts, time_step_target)
            
            if substeps>max_subsetps and not warned:
                _LOG.error("adaptive step results in more than %d substeps.", max_subsetps)
                warned = True
                #LOG.error("adaptive step would result in more than %d substeps.", max_subsetps)
                #return False
            
            sim_ok = PISO_split_step(domain, steps=1, time_step=ts, STOP_FN=STOP_FN, **split_step_kwargs)
            substep += 1
            
            if not sim_ok or STOP_FN():
                return False
    _LOG.info("Adaptive time step %.03e used %d substeps.", time_step, substep)
    return True

def add_pressure_ts_norm(prep_fn):
    _LOG.info("Using normalized pressure!")
    def pressure_ts_norm_1(domain, time_step, **kwargs):
        domain.setPressureRHSdiv(domain.pressureRHSdiv*time_step.cuda())
        domain.UpdateDomainData()
    def pressure_ts_norm_2(domain, time_step, **kwargs):
        domain.setPressureResult(domain.pressureResult/time_step.cuda())
        domain.UpdateDomainData()
    prepend_prep_fn(prep_fn, "POST_PRESSURE_SETUP", pressure_ts_norm_1)
    prepend_prep_fn(prep_fn, "POST_PRESSURE_NON_ORTHO", pressure_ts_norm_2)

def runSim(domain:PISOtorch.Domain, iterations:int, *, time_step:float=1.0, substeps:int=1, corrector_steps:int=2, vel_pre_steps:int=0, density_viscosity:float=None,
           static:bool=False, prep_fn=None, pressure_use_BiCG:bool=False, scipy_solve_advection:bool=False, scipy_solve_pressure:bool=False,
           advection_tol:float=None, pressure_tol:float=None, convergence_tol:float=None, 
           advect_non_ortho_steps:int=1, pressure_non_ortho_steps:int=1, pressure_return_best_result:bool=False,
           velocity_corrector=0,
           log_dir:str=None, log_interval:int=0, log_images:bool=True, norm_vel:bool=False, block_layout=None, output_mode3D:str="slice", log_fn=None,
           output_resampling_coords=None, output_resampling_shape=10,
           save_domain_name:str=None, STOP_FN=lambda: False):
    
    # time_step: physical time to pass per iteration and substep
    # substeps: how many piso steps to make per iteration
    # corrector_steps: number of corrector steps in the PISO algorithm
    # static: only advect the passive scalar
    
    # tolerances:
    # - advection_tol: tolerance for advection BiCG convergence (residual)
    # - pressure_tol: tolerance for pressure CG convergence (residual)
    # - convergence_tol: tolerance for simulation convergence (difference between consecutive steps)
    _LOG.warning("PISOtorch_sim.runSim() is deprecated. Use PISOtorch_simulation.Simulation().run() instead.")
    if log_interval>0 and log_images and log_dir is None:
        raise ValueError("need to specify log/output directory")
    _LOG.info("Starting sim with %d iterations, output in %s\n%s\nBlocks:", iterations, log_dir or "NONE", domain)
    for blockIdx in range(domain.getNumBlocks()):
        _LOG.info(str(domain.getBlock(blockIdx)))
    with SAMPLE("runSim"): #(log_dir if log_dir is not None else "runSim"):
        #for blockIdx in range(domain.getNumBlocks()):
        #    _LOG.info(str(domain.getBlock(blockIdx)))
        
        sim_ok = True
        time_step_target = time_step
        #substeps = 1
        max_mag = 1
        max_mag_temp = max_mag
        CFL_cond = 0.8
        adaptive_step = False
        
        if substeps>0:
            pass # just fixed substeps. 1 iteration with have physical time = time_step*substeps.
        elif substeps==-1:
            # compute max time step for each iteration/substep based on current velocity. 1 iteration with have physical time = time_step.
            adaptive_step = True
        elif substeps==-2:
            # compute max time step based on initial conditions, then keep it constant. 1 iteration with have physical time = time_step.
            time_step, substeps = get_max_time_step(domain, time_step, CFL_cond, with_transformations=True)
            _LOG.info("Setting time step to %.02e, substeps to %d based on initial conditions.", time_step, substeps)
        else:
            raise ValueError("Invalid substeps")
        
        time_step = time_step *torch.ones([1], dtype=domain.getBlock(0).velocity.dtype, device=cpu_device)
        
        if prep_fn is None:
            prep_fn = {}
        elif not isinstance(prep_fn, dict):
            prep_fn = {"PRE": prep_fn}
        
        #DEBUG pressure stability timestep influence
        if False:
            add_pressure_ts_norm(prep_fn)
        
        out_it = 1

        out_dir = log_dir
        os.makedirs(out_dir, exist_ok=True)
        
        vel_exr=False #not static

        sim_info = SimInfo()
        
        if log_images:
            #save_transform_exr(domain, out_dir, "t", 0, layout=block_layout)
            save_domain_images(domain, out_dir, 0, layout=block_layout, norm_p=True, max_mag=max_mag, mode3D=output_mode3D, vel_exr=vel_exr,
                               vertex_coord_list=output_resampling_coords, resampling_out_shape=output_resampling_shape)
        for it in range(1, iterations+1):
            with SAMPLE("Iteration"):
                #LOG.info("It: %d/%d", it, iterations)
                log = log_interval>0 and (it%log_interval)==0
                advect = it>vel_pre_steps
                
                _LOG.info("It: %d/%d, advect:%s, substeps:%s, timestep:%f", it, iterations, advect, "adaptive" if adaptive_step else substeps, time_step_target)
                with SAMPLE("simIt"):
                    if static:
                        sim_ok = advect_static(domain, steps=substeps, time_step=time_step, advect_non_ortho_steps=advect_non_ortho_steps,
                            scipy_solve_advection=scipy_solve_advection, prep_fn=prep_fn, STOP_FN=STOP_FN)
                    elif not advect:
                        sim_ok = PISO_div_free_v2(domain, steps=substeps, time_step=time_step,
                            pressure_non_ortho_steps=pressure_non_ortho_steps, pressure_return_best_result=pressure_return_best_result,
                            scipy_solve_pressure=scipy_solve_pressure, pressure_use_BiCG=pressure_use_BiCG, pressure_tol=pressure_tol,
                            velocity_corrector=velocity_corrector, prep_fn=prep_fn, STOP_FN=STOP_FN)
                    elif adaptive_step:
                        sim_ok = PISO_adaptive_step_v2(domain, time_step=time_step_target, CFL_cond=CFL_cond,
                            corrector_steps=corrector_steps, density_viscosity=density_viscosity, prep_fn=prep_fn, 
                            scipy_solve_advection=scipy_solve_advection, scipy_solve_pressure=scipy_solve_pressure, pressure_use_BiCG=pressure_use_BiCG, 
                            advection_tol=advection_tol, pressure_tol=pressure_tol,
                            advect_non_ortho_steps=advect_non_ortho_steps, pressure_non_ortho_steps=pressure_non_ortho_steps, pressure_return_best_result=pressure_return_best_result,
                            velocity_corrector=velocity_corrector, convergence_tol=convergence_tol, STOP_FN=STOP_FN, sim_info=sim_info)
                    else:
                        sim_ok = PISO_split_step(domain, steps=substeps, corrector_steps=corrector_steps, time_step=time_step, density_viscosity=density_viscosity,
                            scipy_solve_advection=scipy_solve_advection, scipy_solve_pressure=scipy_solve_pressure, pressure_use_BiCG=pressure_use_BiCG,
                            advection_tol=advection_tol, pressure_tol=pressure_tol,
                            advect_non_ortho_steps=advect_non_ortho_steps, pressure_non_ortho_steps=pressure_non_ortho_steps, pressure_return_best_result=pressure_return_best_result,
                            velocity_corrector=velocity_corrector, prep_fn=prep_fn, convergence_tol=convergence_tol, STOP_FN=STOP_FN, sim_info=sim_info)
                    # if not sim_ok:
                        # break
                
                with SAMPLE("vel mag"):
                    max_mag_temp = tensor_as_np(domain.getMaxVelocityMagnitude(False))
                    #_LOG.info("Max vel magnitude: %.03e ", max_mag_temp, max_mag_temp_old)
                    if log: max_vel_temp = tensor_as_np(domain.getMaxVelocity(False))
                    #max_vel_transformed = domain.getMaxVelocity(False, True).cpu().numpy()
                    #_LOG.info("Max vel: %.03e, with bounds %.03e, transformed block 0 %.03e", max_vel_temp, domain.getMaxVelocity(True).cpu().numpy(), max_vel_transformed)
                    if np.isnan(max_mag_temp):
                        _LOG.warning("NaN encountered in velocity, stopping")
                        sim_ok = False
                        break
                    elif not advect and max_mag_temp*time_step_target>CFL_cond:
                        _LOG.warning("CFL condition violation.")
                        sim_ok = False or not advect
                        #break
                    if not advect or norm_vel: # or True:
                        max_mag = max_mag_temp * 1.05
                
                
                if log:
                    with SAMPLE("vel div"):
                        vel_div = PISOtorch.ComputeVelocityDivergence(domain).detach()
                        vel_div_abs = torch.abs(vel_div)
                    with SAMPLE("p stats"):
                        p = domain.pressureResult.detach()
                        p_mean = torch.mean(p).cpu().numpy()
                        p_min = torch.min(p).cpu().numpy()
                        p_max = torch.max(p).cpu().numpy()
                        del p
                    with SAMPLE("output"):
                        _LOG.info("%d/%d Stats:\nVelocity: max=%.03e, max mag=%.03e\nPressure: mean=%.03e, min=%.03e, max=%.03e\nDivergence: mean=%.03e, min=%.03e, max=%.03e, total=%.03e", it, iterations,
                            max_vel_temp, max_mag_temp,
                            p_mean, p_min, p_max,
                            torch.mean(vel_div_abs).cpu().numpy(),torch.min(vel_div_abs).cpu().numpy(), torch.max(vel_div_abs).cpu().numpy(), torch.sum(vel_div_abs).cpu().numpy())
                        if log_images: save_domain_images(domain, out_dir, it, layout=block_layout, norm_p=True, max_mag=max_mag, mode3D=output_mode3D, vel_exr=vel_exr,
                                                          vertex_coord_list=output_resampling_coords, resampling_out_shape=output_resampling_shape)
                        if log_fn is not None:
                            log_fn(domain=domain, out_dir=out_dir, it=it, out_it=out_it)
                        out_it += 1
                
                if STOP_FN() or (not sim_ok):
                    break
    
    if save_domain_name is not None:
        domain_path = os.path.join(out_dir, save_domain_name)
        _LOG.info("sim saved as: %s", domain_path)
        save_domain(domain, domain_path)
