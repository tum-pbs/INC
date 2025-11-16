import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path


def find_run_dir(run_id, task="KS", model_type="UNet", correction_term="INC", down_resolution=64):
    """
    Locate the run directory using run_id and model parameters.
    
    Args:
        run_id: Unique identifier (timestamp) for the run
        task: Task name (KS, Burgers, etc.)
        model_type: Model architecture type
        correction_term: Correction term type
        down_resolution: Resolution
        
    Returns:
        Full path to the run directory
    """
    method_base = f"{model_type}_Corr-{correction_term}"
    base_path = Path(f"INC_Data/{task}/Results/{method_base}/Res_{down_resolution}")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    # Find directories matching the run_id
    run_dirs = sorted([p for p in base_path.glob("*") if run_id in p.name])
    
    if not run_dirs:
        # List available directories for helpful error message
        available = [p.name for p in base_path.glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"No run directory with run_id '{run_id}' found in {base_path}\n"
            f"Available directories: {available[:5]}..."  # Show first 5
        )
    
    if len(run_dirs) > 1:
        print(f"Warning: Multiple directories match run_id '{run_id}': {[p.name for p in run_dirs]}")
        print(f"Using: {run_dirs[0]}")
    
    return str(run_dirs[0])

def create_dataframe(run_id, model_type="UNet", correction_term="INC", down_resolution=64, task="KS", steps=None):
    """
    Create a dataframe from test results for a given run.
    
    Args:
        run_id: Unique identifier (timestamp) for the run
        model_type: Model architecture type
        correction_term: Correction term type
        down_resolution: Resolution
        task: Task name
        steps: Optional filter for specific step count
        
    Returns:
        df: DataFrame with predictions and targets
        data_path: Path to the loaded data file
    """
    # Locate the run directory
    run_dir = find_run_dir(run_id, task=task, model_type=model_type, 
                           correction_term=correction_term, down_resolution=down_resolution)
    
    # Parse parameters from directory name
    dir_name = os.path.basename(run_dir)
    params = {
        'task': task,
        'model_type': model_type,
        'correction_term': correction_term,
        'down_resolution': down_resolution,
    }
    
    # Parse additional parameters from directory name
    for param in dir_name.split('_'):
        if param.startswith('mstep'):
            params['mstep'] = int(param[5:])
        elif param.startswith('dt'):
            params['dt'] = float(param[2:])
        elif param.startswith('lr'):
            params['lr'] = float(param[2:])
        elif param.startswith('wd'):
            params['wd'] = float(param[2:])
    
    # Load test results
    test_dir = os.path.join(run_dir, 'test_results')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test results directory not found: {test_dir}")
    
    results_files = [f for f in os.listdir(test_dir) if f.endswith('.pth')]
    
    if not results_files:
        raise FileNotFoundError(f"No .pth files found in {test_dir}")
    
    # Select the best file based on loss (lowest)
    if steps is not None:
        # Filter by steps if specified
        matching_files = [f for f in results_files if str(steps) in f]
        if not matching_files:
            print(f"Warning: No files found with steps={steps}, using best file instead")
            best_file = min(results_files, key=lambda x: float(x.split('_')[-1].replace('.pth', '').replace('_', '.')))
        else:
            best_file = matching_files[0]
    else:
        # Find file with lowest loss
        best_file = min(results_files, key=lambda x: float(x.split('_')[-1].replace('.pth', '').replace('_', '.')))
    
    data_path = os.path.join(test_dir, best_file)
    print(f"Loading data from: {data_path}")
    
    data = torch.load(data_path)
    
    # Extract predictions and targets
    predictions = torch.cat(data['predictions'], dim=0).cpu().numpy().astype(np.float32)
    targets = torch.cat(data['targets'], dim=0).cpu().numpy().astype(np.float32)
    
    # Create DataFrame
    df = pd.DataFrame({
        **{k: [v] * len(predictions) for k, v in params.items()},
        'prediction': list(predictions),
        'target': list(targets)
    })
    
    return df, data_path

def load_results(run_ids, task="KS", model_configs=None, steps=None):
    """
    Load and process multiple run results.
    
    Args:
        run_ids: List of run IDs (timestamps)
        task: Task name (KS, Burgers, etc.)
        model_configs: List of dicts with model_type, correction_term, down_resolution
                      If None, assumes all runs have same default config
        steps: Optional filter for specific step count
        
    Returns:
        Combined DataFrame with all results
        Path to last loaded data file
    """
    all_data = []
    data_path = None
    
    if model_configs is None:
        # Default config for all runs
        model_configs = [{"model_type": "UNet", "correction_term": "INC", "down_resolution": 64}] * len(run_ids)
    
    for run_id, config in zip(run_ids, model_configs):
        print(f"\nLoading run: {run_id}")
        print(f"Config: {config}")
        
        # Load data and parameters
        df, data_path = create_dataframe(
            run_id=run_id,
            task=task,
            steps=steps,
            **config
        )
        
        # Create method label
        df['method'] = (df['model_type'] + "_" + 
                       df['correction_term'].fillna('Pred') + "_" + 
                       df['down_resolution'].astype(str))
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df, data_path

def get_torch(df):
    df = df.reset_index(drop=True) 
    B = len(df["target"])
    T, D = df["target"][0].shape
    target = torch.tensor(np.stack(df["target"].values), dtype=torch.float32).reshape(B, T, D)
    prediction = torch.tensor(np.stack(df["prediction"].values), dtype=torch.float32).reshape(B, T, D)
    return {"target": target, "prediction": prediction}

def get_method_dict(df):
    method_columns = df.value_counts("method").keys().tolist()
    method_dict = {}
    for i in range(len(method_columns)):
        method_dict[method_columns[i]] = get_torch(df.query(f"method == '{method_columns[i]}'"))
    return method_dict

def calculate_energy_spectrum(velocity):
    """
    Computes the energy spectrum and physical wavenumbers from velocity field.
    
    Args:
        velocity (numpy.ndarray): Velocity field with shape [B, T, D] or [T, D]
        
    Returns:
        energy_spectrum (numpy.ndarray): Shape [B, T, spectrum_len] or [T, spectrum_len]
        k_phys (numpy.ndarray): Physical wavenumbers, shape [spectrum_len]
    """
    # Handle different input shapes
    if velocity.ndim == 2:
        velocity = np.expand_dims(velocity, 0)  # Add batch dimension if missing
    
    B, T, D = velocity.shape
    
    # Convert to torch tensor for easier FFT operations
    velocity_tensor = torch.tensor(velocity, dtype=torch.float32)
    
    # Compute FFT and energy
    u_hat = torch.fft.fft(velocity_tensor, dim=-1)
    # energy = (u_hat.real**2 + u_hat.imag**2) / (D**2)  # Normalized energy
    energy = (u_hat.real**2 + u_hat.imag**2)

    
    # Calculate spectrum length and initialize storage
    spectrum_len = D//2 + 1 if D%2 == 0 else (D+1)//2
    energy_spectrum = energy[..., :spectrum_len].clone()
    
    # Handle symmetry for real signals
    if D % 2 == 0:
        energy_spectrum[..., 1:-1] *= 2  # Exclude DC and Nyquist
    else:
        energy_spectrum[..., 1:] *= 2    # Exclude DC
    
    # Compute physical wavenumbers
    k_indices = torch.arange(spectrum_len, dtype=torch.float32)
    k_phys = 2 * torch.pi * k_indices / torch.ones(B,1)
    return energy_spectrum, k_phys

def plot_2d(original_df, L=16, batch=0, num_times=10, time_start=0, time_end=None,save_dir=None):
    # Get list of unique methods
    method_columns = original_df.value_counts("method").keys().tolist()
    num_methods = len(method_columns)
    colors = plt.cm.RdBu(np.linspace(0, 1, num_times))
    
    # Create subplots: 1 row, +2 columns for each method (prediction and difference)
    fig, axs = plt.subplots(
        1, 2 * num_methods + 1,
        figsize=(2 * (2 * num_methods + 1), 5),  # Adjust width for more columns
        sharex=False, sharey=False,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
        squeeze=False
    )
    
    dt = original_df.iloc[batch]["dt"]
    T = original_df.iloc[batch]["prediction"].shape[0]
    
    # Determine the time window
    if time_end is None or time_end > T:
        time_end = T
    
    desired_times = np.linspace(time_start, time_end, num=num_times)
    
    # --- First Column: imshow for Target ---
    target = original_df.iloc[batch]["target"][time_start:time_end]  # (T, D)
    x = np.linspace(0, L, target.shape[1])
    times = np.clip((desired_times / dt).astype(int), 0, T-1)
    
    ax_target_im = axs[0, 0]
    # Use extent to set x-axis from 0 to L and y-axis from 0 to T*dt (time)
    im_target = ax_target_im.imshow(target, aspect='auto', cmap="RdBu", extent=[0, L, time_start, time_end], origin='lower', vmin=-2, vmax=2)
    ax_target_im.set_xlabel("Target", fontsize=16)
    ax_target_im.set_xlim(0, L)
    ax_target_im.set_xticks([0, L/2])
    ax_target_im.set_ylabel("Times(s)", fontsize=16)
    ax_target_im.tick_params(axis='both', labelsize=14)
    
    # Set y-ticks to dt*2*steps/11
    y_ticks = np.linspace(time_start, time_end, num=11)
    y_tick_labels = [f"{dt*2*t:.2f}" for t in y_ticks]
    ax_target_im.set_yticks(y_ticks)
    ax_target_im.set_yticklabels(y_tick_labels)
    
    # --- For each method: Prediction and Difference ---
    for i, method in enumerate(method_columns):
        df = original_df.query(f"method == '{method}'")
        pred = df.iloc[batch]["prediction"][time_start:time_end]
        mse = df.iloc[batch]["mse"]
        diff = target - pred  # Calculate the difference
        
        # imshow plots for prediction
        ax_im = axs[0, 2 * i + 1]
        ax_im.imshow(pred, aspect='auto', cmap="RdBu", extent=[0, L, time_start, time_end], origin='lower', vmin=-2, vmax=2)
        ax_im.set_xlim(0, L)
        ax_im.tick_params(axis='both', labelsize=14)
        ax_im.set_xlabel(method, fontsize=16)
        ax_im.set_xlim(0, L)
        ax_im.set_xticks([0, L/2, L])
        ax_im.set_yticks([])
        
        # imshow plots for difference
        ax_diff = axs[0, 2 * i + 2]
        ax_diff.imshow(diff, aspect='auto', cmap="RdBu", extent=[0, L, time_start, time_end], origin='lower', vmin=-2, vmax=2)
        ax_diff.set_xlim(0, L)
        ax_diff.tick_params(axis='both', labelsize=14)
        ax_diff.set_xlabel(f"Diff{mse:4f}", fontsize=14)
        ax_diff.set_xlim(0, L)
        ax_diff.set_xticks([0, L/2, L])
        ax_diff.set_yticks([])

    # Colorbar for the target plot
    cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.375])
    cbar = fig.colorbar(im_target, cax=cbar_ax, orientation='vertical')
    cbar.set_label("u(x,t)", fontsize=16)
    save_dir = save_dir + f"/2d_fig/"
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_dir + f"frame{batch}.png" if save_dir is not None else None
    if save_name is not None:
        plt.savefig(save_name)
    plt.close()

def plot_energy(energy_spectra, k_values, batch_idx, method, save_dir=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    E_target = energy_spectra[method][batch_idx].mean(dim=0).cpu().numpy()
    E_pred = energy_spectra[f"{method}_pred"][batch_idx].mean(dim=0).cpu().numpy()
    k = k_values[method][batch_idx].cpu().numpy()
    ax.plot(k, E_target, 'o', label='Target', markersize=4)
    ax.plot(k, E_pred, '-', label=f'{method}')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-8)
    ax.set_xlabel('Wavenumber (k)', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    save_dir = save_dir + f"/energy_fig/"
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_dir + f"batch{batch_idx}.png" if save_dir is not None else None
    if save_name is not None:
        plt.savefig(save_name)
    plt.close()

def main(args):
    """
    Main function to load results and generate plots.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load results for specified run IDs
    run_ids = args.model_id  # Can be a list of run IDs
    
    # You can specify different configs for each run if needed
    # For now, using default config
    model_configs = None  # Will use default UNet_Corr-INC_Res64
    
    # If you want to specify different configs for different runs:
    # model_configs = [
    #     {"model_type": "UNet", "correction_term": "INC", "down_resolution": 64},
    #     {"model_type": "FNO", "correction_term": "INC", "down_resolution": 64},
    # ]
    
    df_base, data_path = load_results(
        run_ids=run_ids,
        task=args.task,
        model_configs=model_configs,
        steps=args.steps
    )
    
    # Create save directory from data path
    save_dir = data_path.replace(".pth", "_fig/")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving plots to: {save_dir}")
    
    # Calculate MSE
    df_base['mse'] = df_base.apply(lambda row: np.mean((row['prediction'] - row['target']) ** 2), axis=1)
    
    # Get method dictionary
    method_dict = get_method_dict(df_base)
    
    # Calculate energy spectra
    energy_spectra = {}
    k_values = {}
    for method in method_dict.keys():
        energy_spectra[method], k_values[method] = calculate_energy_spectrum(method_dict[method]["target"])
        energy_spectra[f"{method}_pred"], k_values[f"{method}_pred"] = calculate_energy_spectrum(method_dict[method]["prediction"])

    # Generate plots for each method
    for method in method_dict.keys():
        df = df_base.query(f"method == '{method}'")
        print(f"\nGenerating plots for method: {method}")
        for i in range(len(df)):
            if i % 100 == 0:
                print(f"  Processing batch {i}/{len(df)}")
                plot_2d(df, batch=i, num_times=1000, time_start=0, time_end=None, save_dir=save_dir)
                plot_energy(energy_spectra, k_values, i, method, save_dir=save_dir)
    
    print(f"\nDone! All plots saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot KS equation analysis.')
    parser.add_argument('--model_id', type=str, nargs='+', required=True, 
                       help='One or more run IDs (timestamps) to analyze')
    parser.add_argument('--task', type=str, default='KS', 
                       help='Task name (default: KS)')
    parser.add_argument('--steps', type=int, default=None, 
                       help='Filter for specific step count (optional)')
    args = parser.parse_args()
    main(args)