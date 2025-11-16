import os
import sys
cudaID = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = cudaID

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from config import config_2d as config
from data.DomainManager import DomainData, BFSDomainManager
from lib.data.BlocksHandling import split_into_blocks, get_block_shape_by_grid, resample_data
import argparse


def load_results(run_id, task="BFS",  forward_filter=None):
    """
    Load test results based on a unique run_id.

    Parameters:
        run_id (str): Unique identifier for the run (timestamp)
        task (str): Task name (default: BFS)
        forward_filter (optional): Used to filter filenames if provided

    Returns:
        combined_df (pd.DataFrame): The concatenated dataframe of test results
        combined_correction_input_df (pd.DataFrame or None): The concatenated dataframe of correction/input results if available
        test_data_dir (str): The full path to the test_results directory
    """
    # Build path using the new structure
    base_path = Path(f"INC_Data/{task}/Results/")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    # Find run directory matching the run_id
    run_dirs = sorted([p for p in base_path.glob(f"**/*{run_id}*") if p.is_dir()])
    
    if not run_dirs:
        available = [p.name for p in base_path.glob("*/*") if p.is_dir()]
        raise FileNotFoundError(
            f"No run directory with run_id '{run_id}' found in {base_path}\n"
            f"Available directories (first 5): {available[:5]}..."
        )
    
    if len(run_dirs) > 1:
        print(f"Warning: Multiple directories match run_id '{run_id}': {[p.name for p in run_dirs]}")
        print(f"Using: {run_dirs[0]}")
    
    test_data_dir = run_dirs[0] / "test_results"
    
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test results directory not found: {test_data_dir}")
    
    print(f"Test data directory: {test_data_dir}")

    # Build the template for test result pickle files
    if forward_filter is not None:
        pattern = f"*results*Start*_Forward{forward_filter}.pkl"
    else:
        pattern = "*results*Start*.pkl"

    file_list = list(test_data_dir.glob(pattern))
    print(f"Found {len(file_list)} files")

    # Function to extract the start step from the filename
    def extract_start_step(filepath):
        filename = filepath.name
        if "_Start" in filename:
            try:
                return int(filename.split("_Start")[1].split("_")[0])
            except Exception:
                return -1
        return -1

    sorted_files = sorted(file_list, key=extract_start_step)
    combined_results = []
    correction_input_results = []

    # Loop over the files, load and process each pickle file
    for file_path in sorted_files:
        test_results = pd.read_pickle(file_path)
        start_step = extract_start_step(file_path)
        test_results_expanded = pd.DataFrame(test_results.iloc[0]["data"])
        test_results_expanded["file"] = file_path.name
        test_results_expanded["start_step"] = start_step
        test_results_expanded["step"] = range(len(test_results_expanded))
        combined_results.append(test_results_expanded)
        
        # If the test_results also include 'correction' and 'input', handle them
        if "correction" in test_results and "input" in test_results:
            if len(test_results.iloc[0]["correction"]) == len(test_results.iloc[0]["data"]):
                test_results_expanded['correction'] = test_results.iloc[0]['correction']
                test_results_expanded['input'] = test_results.iloc[0]['input']
            else:
                # Handle cases where corrections/inputs are organized separately
                correction = test_results.iloc[0]["correction"]
                input_data = test_results.iloc[0]["input"]
                correction_input_df = pd.DataFrame({
                    "step": range(len(correction)),
                    "correction": correction,
                    "input": input_data,
                    "file": file_path.name,
                    "start_step": start_step,
                })
                correction_input_results.append(correction_input_df)

    combined_df = pd.concat(combined_results, ignore_index=True)
    if correction_input_results:
        combined_correction_input_df = pd.concat(correction_input_results, ignore_index=True)
        return combined_df, combined_correction_input_df, str(test_data_dir)
    else:
        return combined_df, None, str(test_data_dir)

def get_resampled_velocity(target, data,sim_params,mask=True):
    domain_manager = BFSDomainManager(**sim_params.__dict__)
    layout=domain_manager.get_config(domain_manager.get_all_keys()[0]).layout
    grids=domain_manager.get_config(domain_manager.get_all_keys()[0]).grids
    block_shape=get_block_shape_by_grid(grids,layout)
    res_y=data.iloc[0][target].shape[-2]
    ratio = int(res_y/(sim_params.H))
    res_x=(sim_params.l+sim_params.L+sim_params.buffer_list[0])*ratio
    resmapled_list=[]
    for i in range(len(data)):
        vel_list = split_into_blocks(torch.tensor(data.iloc[i][target]),layout,block_shape)
        vel_list=[torch.tensor(velocity,dtype=torch.float32).cuda() for velocity in vel_list]
        resampled=resample_data(vel_list,grids,res_x,res_y,fill_max_steps=8)[0].cpu().numpy()
        resmapled_list.append(resampled)
    result = np.array(resmapled_list)
    if mask:
        print(f"Mask area: {int(sim_params.h*result.shape[-2]/(sim_params.H))}x{int(sim_params.l*result.shape[-1]/(sim_params.l+sim_params.L+sim_params.buffer_list[0]))}")
        print(f"Shape: {result.shape}")
        result[:,:,0:int(sim_params.h*result.shape[-2]/(sim_params.H)),0:int(sim_params.l*result.shape[-1]/(sim_params.l+sim_params.L+sim_params.buffer_list[0]))]=0
    return result

def plot_stream_plot(velocity_res, sim_params, name, ax=None, cmap='coolwarm'):
    velocity_res_average = np.mean(velocity_res, axis=0)
    # Compute x_positions for the domain
    x_positions = np.linspace(-sim_params.l/sim_params.h, (sim_params.L+sim_params.buffer_list[0])/sim_params.h, velocity_res_average.shape[2])
    # Only keep x_positions where x <= 30
    right = np.abs(x_positions - 33).argmin()
    velocity_res_average = velocity_res_average[:, :, :right+1]
    x_positions = x_positions[:right+1]
    u_res = np.sqrt(velocity_res_average[0]**2 + velocity_res_average[1]**2)
    y_positions = np.linspace(0, sim_params.H/sim_params.h, u_res.shape[0])
    uniform_X, uniform_Y = np.meshgrid(x_positions, y_positions)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.streamplot(
        uniform_X, uniform_Y,
        velocity_res_average[0], velocity_res_average[1],
        density=[4, 0.85], color=u_res, cmap=cmap,
        arrowsize=0.75, linewidth=1.5, norm=None
    )
    # Add grey filled rectangle for the region ((-sim_params.l,0),(0,0),(0,sim_params.h),(-sim_params.l,sim_params.h))
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (-sim_params.l/sim_params.h, 0),  # (x, y) lower left
        sim_params.l/sim_params.h,        # width
        sim_params.h/sim_params.h,        # height
        linewidth=0, edgecolor=None, facecolor='grey', alpha=0.3, zorder=1
    )

    ax.add_patch(rect)
    ax.set_title(f"{name}",fontsize=22)
    ax.set_ylim(0, sim_params.H/sim_params.h)
    ax.set_xlim(x_positions[0], x_positions[-1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    if "INC" in name:
        ax.set_title(f"{name}", fontsize=22, bbox=dict(facecolor='None', edgecolor='black', boxstyle='round'))
    return ax


def main(args):
    """
    Main function to load BFS results and generate plots.
    
    Args:
        args: Parsed command-line arguments
    """
    # Extract parameters from args
    run_id = args.model_id
    forward_step = args.forward_step
    Re = args.Re
    
    print(f"\nLoading results for run_id: {run_id}")
    print(f"Forward steps: {forward_step}, Re: {Re}")
    
    # Load main results
    original_velocity, _, test_data_dir = load_results(
        run_id, 
        task="BFS",
        forward_filter=forward_step
    )
    
    # Try to load no-model baseline for comparison
    try:
        print("\nAttempting to load no-model baseline...")
        no_model_velocity, _, _ = load_results(
            "241209-211752",
            forward_filter=forward_step
        )
        load_No_model = True
        print("Successfully loaded no-model baseline")
    except Exception as e:
        print(f"Warning: Could not load no-model baseline: {e}")
        load_No_model = False
    
    # Create save directory
    save_dir = os.path.join(test_data_dir, "test_analysis")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving plots to: {save_dir}")
    
    # Generate plots for each starting point
    for starting_point in original_velocity['start_step'].unique():
        print(f"\nProcessing starting point: {starting_point}")
        
        resampled_velocity_ref = get_resampled_velocity(
            "ref_u",
            original_velocity.query(f"start_step=={starting_point}")[:forward_step],
            config.BFSSimParams(Re=Re, s=1, downsample_factor=4),
            mask=True
        )
        
        resampled_velocity_res = get_resampled_velocity(
            "res_u",
            original_velocity.query(f"start_step=={starting_point}")[:forward_step],
            config.BFSSimParams(Re=Re, s=1, downsample_factor=4),
            mask=True
        )
        
        resampled_velocity_no_model = get_resampled_velocity(
            "res_u",
            no_model_velocity.query(f"start_step=={starting_point}")[:forward_step],
            config.BFSSimParams(Re=Re, s=1, downsample_factor=4),
            mask=True
        ) if load_No_model else np.zeros_like(resampled_velocity_ref)
        
        results_list = [
            resampled_velocity_ref[:forward_step], 
            resampled_velocity_no_model[:forward_step], 
            resampled_velocity_res[:forward_step]
        ]

        labels = ["Ref.", "No Model", "INC"]
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i, (result, label) in enumerate(zip(results_list, labels)):
            plot_stream_plot(
                result,
                config.BFSSimParams(Re=Re, s=1, downsample_factor=4),
                name=label,
                ax=axes[i],
            )
        
        plot_path = os.path.join(save_dir, f"Stream_compare{Re}_{starting_point}.svg")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved: {plot_path}")
    
    print(f"\nDone! All plots saved to: {save_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot BFS analysis.')
    parser.add_argument('--model_id', type=str, required=True, 
                       help='Run ID (timestamp)')
    parser.add_argument('--forward_step', type=int, default=1200,
                       help='Number of forward steps (default: 1200)')
    parser.add_argument('--Re', type=int, default=1400,
                       help='Reynolds number (default: 1400)')
    args = parser.parse_args()
    main(args)