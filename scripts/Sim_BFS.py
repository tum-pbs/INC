import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def make_dataset(steps):
    import lib.util.domain_io as domain_io
    def base_log_fn(domain,out_dir,out_it,**kwargs):
        domain_io.save_domain(domain,os.path.join(out_dir,"BFSdomain_%04d"%out_it))
    
    for (config_key,domainData) in domain_manager.domain_dict.items():
        LOG.info(f"Running simulation with configuration key: {config_key}")
        domain=domainData.domain
        prep_fn=domainData.prep_fn
        layout=domainData.layout
        log_dir=os.path.join(run_dir, f"Re{sim_params.Re}_Base{'-'.join(map(str, config_key[0]))}_Step{config_key[1]}_Dt{sim_params.time_step}")
        sim=PISOtorch_simulation.Simulation(domain=domain,time_step=sim_params.time_step,block_layout=layout,prep_fn=prep_fn,substeps=sim_params.substeps,corrector_steps=2, pressure_tol=sim_params.pressure_tol,advect_non_ortho_steps=1, differentiable=False,pressure_non_ortho_steps=1, pressure_return_best_result=True,velocity_corrector="FD", non_orthogonal=False,norm_vel=True,log_dir=log_dir,log_fn=base_log_fn,log_interval=1,save_domain_name="BFSdomain",stop_fn=None)
        sim.run(steps)

if __name__ == "__main__":
    from lib.util.GPU_info import get_available_GPU_id
    import argparse
    parser = argparse.ArgumentParser(description='Run simulations with varying Reynolds numbers and grid refinement factors.')
    parser.add_argument('--Re', type=int, required=True, help='Reynolds numbers')
    parser.add_argument('--s', type=float, default=1, help='Step height')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for the simulation')
    args = parser.parse_args()
    cudaID = str(args.gpu_id)
    cudaID = cudaID or str(get_available_GPU_id(active_mem_threshold=0.8, default=None))
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaID
    import numpy as np
    import torch
    assert torch.cuda.is_available()
    import PISOtorch
    from solvers.solver_2d import PISOtorch_diff # differentiable wrapper for PISO
    from solvers.solver_2d import PISOtorch_simulation # helper for PISO loop
    from lib.util.logging import setup_run, get_logger, close_logging # logging and output
    from data.DomainManager import BFSDomainManager
    from config import config_2d as config
    LOG = get_logger("Main")
    downsample_factor = 4/1
    Res = "512x128"
    # Res = "640x160"
    # Res = "1024x256"
    base_list = [[20,2]]
    # run_dir = setup_run("./INC_Data/BFS/Dataset", name="Sim")
    run_dir = setup_run("./INC_Data/BFS/Dataset/Benchmark/vis_1", name=f"Sim_{Res}")
    # sim_params= config.BFSSimParams(Re=args.Re, s=args.s,base_list=base_list, downsample_factor=downsample_factor)
    sim_params= config.BFSSimParams(Re=args.Re, s=args.s, downsample_factor=downsample_factor)

    sim_params.vis_ratio = 1 
    # sim_params.time_step = 0.05
    LOG.info(f"BFSDomainManager parameters: {sim_params.__dict__}")
    domain_manager = BFSDomainManager(**sim_params.__dict__)
    steps=6000
    make_dataset(steps)
    close_logging()