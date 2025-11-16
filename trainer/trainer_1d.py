"""
Consolidated 1D trainer module combining base trainer classes and helper functions.
This module contains:
- Helper functions for data loading, simulation, regularization
- Base trainer classes for 1D problems
- Correction and Prediction trainer implementations
"""
import os
import re
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import math
from lib.util.plotting import plot_result, plot_correction
from lib.util.logging import setup_logging


# =============================================================================
# Helper Functions
# =============================================================================

def check_dt(train_param, LOG, run_dir):
    """Check and validate dt parameter from run_dir."""
    dt = train_param.dt
    match = re.search(r'_dt([\d\.]+)', run_dir)
    if match:
        dt_model = float(match.group(1))
    else:
        raise ValueError("Could not find dt in run_dir")
    if abs(dt - dt_model) > 1e-6:
        train_param.dt = dt_model
        LOG.info(f"dt mismatch: {dt} (config) vs {dt_model} (run_dir). Using run_dir dt.")
    else:
        LOG.info(f"dt matched: {dt} (config) vs {dt_model} (run_dir)")
    return dt_model


def get_path(resolution, task="Burgers"):
    """
    Search for a simulation file in run_dir (relative to the location of this file)
    that corresponds to the desired spatial resolution.
    """
    if '__file__' in globals():
        base_path = Path(__file__).resolve().parent
    else:
        base_path = Path(os.getcwd())
    debug_dir = base_path / "INC_Data" / task / "Dataset"
    if not debug_dir.exists():
        # If not found, try one directory up (in case of different calling locations)
        debug_dir = base_path.parent / "INC_Data" / task / "Dataset"

    res_str = str(resolution)
    
    # Pattern for an original simulation file, e.g., "Res512_..."
    original_pattern = re.compile(rf"^Res{res_str}(?:_|$)")
    # Pattern for a downsampled simulation file, e.g., "DownRes128_..."
    downsampled_pattern = re.compile(rf"^DownRes{res_str}(?:_|$)")
    path_list = []
    for file in debug_dir.glob("*.pth"):
        if original_pattern.search(file.stem) or downsampled_pattern.search(file.stem):
            path_list.append(str(file))
    if path_list == []:
        raise FileNotFoundError(f"No file found for resolution {resolution} in {debug_dir}")
    path_dict = {"train": None, "valid": None, "test": None, "original": None, "extend": None}
    for item in path_list:
        lower_item = item.lower()  # for case-insensitive matching
        if "train." in lower_item:
            path_dict["train"] = item
        elif "valid." in lower_item:
            path_dict["valid"] = item
        elif "test." in lower_item:
            path_dict["test"] = item
        elif "_extend" in lower_item:
            path_dict["extend"] = item
        else:
            path_dict["original"] = item
    return path_dict


def load_model_dir(train_param, run_id: str, task_dir: str, LOG=None, mode="train"):
    """Load model directory and return run_dir path and exact model path."""
    if run_id is None:
        raise ValueError("run_id must be provided to load a model")
    method_base = f"{train_param.model_type}_Corr-{train_param.correction_term}"
    base_path = Path(f"{task_dir}{method_base}/Res_{train_param.down_resolution}")
    if not base_path.exists():
        # Fallback: try one directory up.
        base_path = Path(f"../{task_dir}{method_base}/Res_{train_param.down_resolution}")
    
    # Resolve symbolic links to get the actual path
    base_path = base_path.resolve()
    
    # Check if base path exists after resolving
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    run_dirs = sorted([p for p in base_path.glob("*") if run_id in p.name and p.is_dir()])
    if not run_dirs:
        # List available directories to help user debug
        available_dirs = [p.name for p in base_path.glob("*") if p.is_dir()]
        error_msg = f"No run directory with run_id '{run_id}' found in {base_path}\n"
        error_msg += f"Available directories: {available_dirs[:5]}"  # Show first 5
        if len(available_dirs) > 5:
            error_msg += f" ... and {len(available_dirs) - 5} more"
        raise FileNotFoundError(error_msg)
    selected_run = run_dirs[0]
    model_dir = selected_run / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    candidates = sorted([f for f in os.listdir(model_dir) if f.startswith("best_model")])
    if not candidates:
        raise FileNotFoundError(f"No candidate models found in: {model_dir}")
    state_dict_path = model_dir / candidates[0]
    if mode == "test":
        setup_logging(os.path.join(selected_run.__str__(), "test_log"))
    LOG.info(f"Loaded model from {state_dict_path}") if LOG else None
    return selected_run.__str__(), state_dict_path


def extract_epoch(filename):
    """Extract epoch number from filename."""
    match = re.search(r'epoch(\d+)', filename)
    if match:
        return match.group(1)
    match = re.search(r'checkpoint_(\d+)', filename)
    if match:
        return match.group(1)
    return None


def init_dataloader(data_dict, train_param):
    """
    Prepare the dataloaders for training, validation, and testing.
    Divide the dataset into training, validation, and testing sets by randomly splitting 
    the indices in Batch dimension, then create dataloaders.
    """
    def dtype_check(data):
        return {k: v.to(train_param.dtype) if torch.is_tensor(v) else v for k, v in data.items()}
    
    dataset = train_param.dataset
    collate_fn = train_param.collate_fn
    train_data = dataset(dtype_check(data_dict["train"]), mstep=train_param.mstep, dt=train_param.dt, starts_gap=train_param.starts_gap)
    valid_data = dataset(dtype_check(data_dict["valid"]), mstep=train_param.valid_step, dt=train_param.dt, starts_gap=train_param.starts_gap)
    test_data = dataset(dtype_check(data_dict["test"]), mstep=train_param.test_steps, dt=train_param.dt, starts_gap=train_param.starts_gap)
    
    train_loader = DataLoader(train_data, batch_size=train_param.batch_size, shuffle=True, collate_fn=collate_fn, generator=torch.Generator().manual_seed(train_param.seed))
    valid_loader = DataLoader(valid_data, batch_size=train_param.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=train_param.test_batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def simulate_rollout(model, solver, u_initial, mstep, correction_term, L=None):
    """
    Unroll the simulation for mstep steps.
    Inputting velocity (in the shape of [B,D]) to the model
    Returns a tuple of two tensors (results and corrections) in the shape of [B, mstep, D] or only results.
    L is the domain_size, for KS now as it's changing
    """
    u_current = u_initial
    B, D = u_initial.shape
    results = torch.zeros((B, mstep, D), device=u_current.device, dtype=u_current.dtype)
    corrections = torch.zeros((B, mstep, D), device=u_current.device, dtype=u_current.dtype) if correction_term else None
    solver.correction_s = None
    model_fn = lambda u: model(u) if L is None else model(u, L)
    solver_step = lambda u: solver.step(u) if L is None else solver.step(u, L)
    
    for step in range(mstep):
        correction = None
        if correction_term == "INC":
            correction = model_fn(u_current)
            solver.correction_s = correction
            u_next = solver_step(u_current)
        elif correction_term is None:
            u_next = solver_step(u_current)
        else:
            raise ValueError(f"Invalid correction term: {correction_term}")
        results[:, step] = u_next
        if correction is not None:
            corrections[:, step] = correction
        u_current = u_next  # shape: [B, D]
    return results, corrections


def init_pred_dataloader(data_dict, train_param):
    """
    Initialize data loaders for training, validation, and testing.
    
    The function processes time-series data by:
    1. Resampling based on the specified dt ratio
    2. Unfolding time dimension into batches using sliding windows
    3. Creating appropriate DataLoaders with task-specific configurations
    
    Input data shape: [B, T, D] (Batch, Time, Dimension)
    Output data shape: x: [B', T_in, D], y: [B', T_out, D]
    where B' = B * (T_total - T_in - T_out + 1) due to sliding window approach
    
    Args:
        data_dict: Dictionary containing train/valid/test trajectory data
        train_param: Parameters for training including window sizes and batch sizes
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    T_in = train_param.T_in
    T_out_train = train_param.mstep
    T_out_valid = train_param.valid_step
    T_out_test = train_param.test_steps
    
    ratio = train_param.dt / train_param.metadata.get("gen_dt", 1.0)
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(f"dt must be a multiple of gen_dt ({train_param.metadata.get('gen_dt')})")
    ratio = int(round(ratio))
    starts_gap = ratio if train_param.starts_gap is None else train_param.starts_gap   
    train_data = data_dict["train"]["trajectories"].to(train_param.dtype)
    valid_data = data_dict["valid"]["trajectories"].to(train_param.dtype)
    test_data = data_dict["test"]["trajectories"].to(train_param.dtype)

    def unfold_time_to_batches(data, T_out, starts_gap=starts_gap):
        B, T_total, D = data.shape
        # Calculate valid start positions
        max_start = T_total - (T_in * ratio + T_out * ratio)
        starts = torch.arange(0, max_start + 1, starts_gap)
        
        # Input indices: [t, t+ratio, t+2ratio, ..., t+(T_in-1)ratio]
        input_indices = starts[:, None] + torch.arange(T_in) * ratio
        
        # Output indices: [t+T_in*ratio, t+(T_in+1)ratio, ..., t+(T_in+T_out-1)ratio]
        output_indices = starts[:, None] + T_in * ratio + torch.arange(T_out) * ratio

        inputs = data[:, input_indices, :]  # [B, num_starts, T_in, D]
        outputs = data[:, output_indices, :]  # [B, num_starts, T_out, D]
        
        # [B*num_starts, T_in, D], [B*num_starts, T_out, D]
        return inputs.reshape(-1, T_in, D), outputs.reshape(-1, T_out, D)

    train_inputs, train_outputs = unfold_time_to_batches(train_data, T_out_train)
    valid_inputs, valid_outputs = unfold_time_to_batches(valid_data, T_out_valid)
    test_inputs, test_outputs = unfold_time_to_batches(test_data, T_out_test)
    
    def create_dataloaders(inputs, outputs, batch_size, is_train=False, domain_size=None):
        if domain_size is not None:
            repeat_factor = inputs.shape[0] // domain_size.shape[0]
            domain_size_expanded = domain_size.repeat_interleave(repeat_factor, dim=0).to(train_param.dtype)
            dataset = torch.utils.data.TensorDataset(inputs, outputs, domain_size_expanded)
        else:
            dataset = torch.utils.data.TensorDataset(inputs, outputs)
            
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    
    if train_param.task == "KS":
        train_L = data_dict["train"]["domain_size"]
        valid_L = data_dict["valid"]["domain_size"]
        test_L = data_dict["test"]["domain_size"]
        
        train_loader = create_dataloaders(train_inputs, train_outputs, train_param.batch_size, True, train_L)
        valid_loader = create_dataloaders(valid_inputs, valid_outputs, train_param.batch_size, False, valid_L)
        test_loader = create_dataloaders(test_inputs, test_outputs, train_param.test_batch_size, False, test_L)
        
    elif train_param.task == "Burgers":
        train_loader = create_dataloaders(train_inputs, train_outputs, train_param.batch_size, True)
        valid_loader = create_dataloaders(valid_inputs, valid_outputs, train_param.batch_size, False)
        test_loader = create_dataloaders(test_inputs, test_outputs, train_param.test_batch_size, False)
        
    else:
        raise ValueError(f"Unsupported task: {train_param.task}")
        
    return train_loader, valid_loader, test_loader


def predict_rollout(model, u, mstep, L=None):
    """
    Herein, the mstep is the number of steps to predict, and ratio is the ratio of the dt 
    of the model to the gen_dt of the original data (as ratio in FNO original code). 
    """
    # step = ratio
    step = 1  # since the data is already sampled by ratio, herein step should always be 1 instead of the ratio
    for t in range(0, mstep, step):
        output = model(u, L) if L != None else model(u)
        u = torch.cat((u[:, 1:, :], output), dim=1)
        pred = output if t == 0 else torch.cat((pred, output), dim=1)
    return pred


def gradient_penalty(model, real_inputs, L=None):
    """Computes gradient penalty for Lipschitz regularization"""
    real_inputs.requires_grad_(True)
    outputs = model(real_inputs, L) if L is not None else model(real_inputs)
    
    # Compute gradients w.r.t inputs
    gradients = torch.autograd.grad(
        outputs=outputs, inputs=real_inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
        only_inputs=True
    )[0]
    
    # Penalize gradients deviating from norm=1
    penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return penalty


def spectral_norm_regularization(model, n_power_iterations=1):
    """Compute spectral norm regularization for model weights."""
    spectral_loss = 0.0
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if isinstance(module, torch.nn.Linear):
                W = module.weight  # [out_features, in_features]
            elif isinstance(module, torch.nn.Conv1d):
                W = module.weight.view(module.weight.size(0), -1)  # [out_channels, in_channels * kernel_size]
            # Power iteration for spectral norm
            u = torch.randn(W.size(1), 1, device=W.device)
            with torch.no_grad():
                for _ in range(n_power_iterations):
                    v = torch.nn.functional.normalize(W @ u, dim=0)
                    u = torch.nn.functional.normalize(W.T @ v, dim=0)
                sigma = torch.sum(v * (W @ u))
            spectral_loss = spectral_loss + sigma ** 2
    return spectral_loss


def check_nan(loss):
    """Check if loss is NaN."""
    if isinstance(loss, torch.Tensor):
        return torch.isnan(loss).any()
    elif isinstance(loss, float):
        return math.isnan(loss)
    else:
        return False


# =============================================================================
# Simulation Initialization Functions
# =============================================================================

def init_BGSim(batch, train_param):
    """
    Initialize simulation parameters and the solver from a batch for Burgers equation.
    Returns the solver, initial condition, and target trajectory.
    """
    from solvers.solver_1d import BurgersSolverTorch
    from config.config_1d import BGSimParams
    
    sim_params = BGSimParams(
        dt=train_param.dt,
        resolution=train_param.metadata['resolution'],
        L=train_param.metadata['L'],
        equation_type='learning',
        batch_size=batch['u_initial'].shape[0],
        adaptive_CFL=train_param.adaptive_CFL
    )
    for key in ['nu', 'A', 'omega', 'phi', 'l']:
        setattr(sim_params, key, batch[key].to(sim_params.device))
    solver = BurgersSolverTorch(sim_params)
    solver.t = batch['t'].unsqueeze(1).to(train_param.device)
    return solver, batch['u_initial'].to(train_param.device), batch['u_target'].to(train_param.device), None


def init_KSSim(batch, train_param):
    """
    Initialize simulation parameters and the solver from a batch for KS equation.
    Returns the solver, initial condition, and target trajectory.
    """
    from solvers.solver_1d import KSSolverTorch
    from config.config_1d import KSSimParams
    
    domain_size = batch.get('domain_size', None)
    if domain_size is not None:
        domain_size = domain_size.to(train_param.device)
    sim_params = KSSimParams(
        dt=train_param.dt,
        domain_size=domain_size,
        batch_size=batch['u_initial'].shape[0],
        time_scheme="ETD1")
    solver = KSSolverTorch(sim_params)
    return solver, batch['u_initial'].to(train_param.device), batch['u_target'].to(train_param.device), domain_size.to(train_param.device)


def get_task_config(args):
    """Get task-specific configuration based on args."""
    from config.config_1d import BGTrainParams, KSTrainParams
    
    common_params = {
        'dt': getattr(args, 'dt', None),
        'mstep': getattr(args, 'mstep', None),
        'correction_term': getattr(args, 'correction_term', None),
        'model_type': getattr(args, 'model_type', None),
        'test_steps': getattr(args, 'test_steps', None),
    }

    if args.task == "Burgers":
        return {
            "train_param": BGTrainParams(
                **common_params,
                down_ratio=args.down_ratio,  # specific to Burgers
                init_sim_fn=init_BGSim
            )
        }
    elif args.task == "KS":
        return {
            "train_param": KSTrainParams(
                **common_params,
                init_sim_fn=init_KSSim
            )
        }
    else:
        raise ValueError(f"Unknown task: {args.task}")


# =============================================================================
# Base Trainer Classes
# =============================================================================

class BaseTrainer1D(ABC):
    """
    Abstract base class for 1D trainer, extended by PredictionTrainer and CorrectionTrainer
    
    Parameters:
        model: The neural network model to train
        optimizer: The optimizer for model training
        scheduler: Learning rate scheduler
        train_param: Training parameters and configuration
        writer: TensorBoard writer or similar logging tool
        LOG: Logging object for text logs
        run_dir: Directory to save outputs
    """
    def __init__(self, optimizer, scheduler, train_param, writer, LOG=None, run_dir=None):
        self.model = train_param.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_param = train_param
        self.writer = writer
        self.LOG = LOG
        self.run_dir = run_dir
        self.loss_fn = train_param.loss_fn
        # Tracking variables
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        self.last_lr = self.train_param.lr
        self.prev_NN_out = None
        self.best_model_state = None

    @abstractmethod
    def _log_outputs(self, output, input_data, target, step, mode):
        """
        Log model outputs.
        
        Parameters:
            output: Model output (predictions or corrections)
            input_data: Input data used for the model
            target: Target/ground truth data
            step: Current step number for logging
            mode: Training mode ('Train', 'Valid', 'Test')
        """
        pass

    @abstractmethod
    def process_batch(self, batch, steps, global_steps=None, mode=None):
        """Process a batch of data (abstract method to be implemented by subclasses)"""
        pass

    def _create_result_plot(self, pred, target, mode, batch=0):
        """
        Create result plot for prediction model.
        
        Parameters:
            pred: Predicted trajectories
            target: Target trajectories
            mode: Training mode ('Train', 'Valid', 'Test')
            batch: Batch index for multi-batch plotting
            
        Returns:
            fig: Matplotlib figure object
        """
        return plot_result(pred, target, self.train_param, mode, batch=batch)
    
    def check_dt(self):
        """Check dt parameter consistency."""
        dt = self.train_param.dt
        match = re.search(r'_dt([\d\.]+)', self.run_dir)
        if match:
            dt_model = float(match.group(1))
        else:
            raise ValueError("Could not find dt in run_dir")
        if abs(dt - dt_model) > 1e-6:
            self.train_param.dt = dt_model
            self.LOG.info(f"dt mismatch: {dt} (config) vs {dt_model} (run_dir). Using run_dir dt.")
        else:
            self.LOG.info(f"dt matched: {dt} (config) vs {dt_model} (run_dir)")
        return dt_model

    def _log_batch(self, batch_idx, pred, target, loss, mode):
        """
        Log batch-level information.
        
        Parameters:
            batch_idx: Current batch index
            pred: Model predictions
            target: Target/ground truth data
            loss: Loss value for the batch
            mode: Training mode ('Train', 'Valid', 'Test')
        """
        self.writer.add_scalar(f"Loss/{mode}_Batch", loss.item(), batch_idx)
        
        if mode == "Train" and batch_idx % 1000 == 0:
            self.LOG.info(f"Batch {batch_idx} loss: {loss.item():.6f}")
            fig = self._create_result_plot(pred, target, mode)
            self.writer.add_figure(f"Rollout/{mode}", fig, global_step=batch_idx)
            plt.close(fig)
        elif mode == "Test":
            self.LOG.info(f"Batch {batch_idx} {mode} loss: {loss.item():.6f}")
            num_plots = min(10, target.shape[0])
            for i in range(0, target.shape[0], max(1, target.shape[0] // num_plots)):
                fig = self._create_result_plot(pred, target, mode, batch=i)
                self.writer.add_figure(f"Rollout/{mode}", fig, global_step=i + batch_idx)
                plt.close(fig)

    def _save_checkpoint(self, epoch, train_loss, val_loss=None):
        """
        Save training checkpoints and best model.
        
        Parameters:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            
        Returns:
            bool: True if early stopping should be triggered, False otherwise
        """
        # Save checkpoint every 5 epochs or at the last epoch
        if check_nan(val_loss):
            self.LOG.info(f"Valid loss is NaN at epoch {epoch}. Skipping checkpoint save.")
            return False
        if epoch % 5 == 0 or epoch == self.train_param.epochs - 1:
            checkpoint_dir = os.path.join(self.run_dir, "models/check")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_{epoch}_{train_loss:.6f}.pth".replace('.', '_')
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, checkpoint_path)
            self.LOG.info(f"Saved checkpoint at epoch {epoch}")

        # Save best model only if validation loss improves
        if val_loss is not None:
            if val_loss < self.best_val_loss:
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                    self.LOG.info(f"Removed previous best model: {self.best_model_path}")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                model_dir = os.path.join(self.run_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                self.best_model_path = os.path.join(model_dir, f"best_model_epoch{epoch}_loss{val_loss:.6f}".replace('.', '_') + ".pth")
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                self.LOG.info(f"New best model saved with loss {val_loss:.6f}")
            else:
                self.patience_counter += 1
                self.LOG.info(f"EarlyStopping counter: {self.patience_counter}/{self.train_param.early_stop_patience}")
                if self.patience_counter >= self.train_param.early_stop_patience:
                    self.LOG.info("Early stopping")
                    return True
        return False

    def train(self, dataloader, epoch):
        """
        Run one epoch of training.
        
        Parameters:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            global_steps = epoch * len(dataloader) + i
            self.optimizer.zero_grad()
            loss, u_pred, u_target, _ = self.process_batch(batch, self.train_param.mstep, global_steps, "Train")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if check_nan(loss):
                self.LOG.info(f"Training loss is NaN at epoch {epoch}, batch {i}. Stop training")
                raise ValueError("Training loss is NaN")
            self._log_batch(global_steps, u_pred, u_target, loss, "Train")
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, epoch):
        """
        Evaluate model on validation data for saving model and learning rate adjustment.
        
        Parameters:
            dataloader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                global_steps = epoch * len(dataloader) + i
                loss, _, _, _ = self.process_batch(batch, self.train_param.valid_step, global_steps, "Valid")
                total_loss += loss.item()
        valid_loss = total_loss / len(dataloader)
        self.scheduler.step(valid_loss)
        self.writer.add_scalar("Loss/Valid", valid_loss, epoch)
        self.LOG.info(f"Validation loss: {valid_loss:.6f}")
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr != self.last_lr:
            self.patience_counter = max(0, self.patience_counter - 5)  # give the modified lr at least 5 more epochs
            self.LOG.info(f"Learning rate changed from {self.last_lr:.6f} to {current_lr:.6f}, reset patience counter to {self.patience_counter}")
            self.last_lr = current_lr
        return valid_loss

    def test(self, dataloader, model_path=None):
        """
        Test the model on test data.
        
        Parameters:
            dataloader: Test data loader
            model_path: Path to model checkpoint (optional)
            
        Returns:
            float: Average test loss
        """
        self.LOG.info("Testing...")
        model_path = self.best_model_path if self.best_model_path is not None and model_path is None else model_path
        self.model.load_state_dict(torch.load(model_path, map_location=self.train_param.device)) if "checkpoint" not in str(model_path) else self.model.load_state_dict(torch.load(model_path, map_location="cuda")["model_state_dict"])
        all_preds, all_targets = [], []
        total_loss, total_samples = 0, 0
        all_domain_size = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                loss, u_pred, u_target, domain_size = self.process_batch(batch, self.train_param.test_steps, i, "Test")
                batch_size = u_target.shape[0]
                total_loss += loss.item() * batch_size
                self._log_batch(i, u_pred, u_target, loss, "Test")
                all_preds.append(u_pred)
                all_targets.append(u_target)
                all_domain_size.append(domain_size)
                total_samples += batch_size
        test_loss = total_loss / total_samples
        test_results_dir = os.path.join(self.run_dir, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)
        results_path = os.path.join(test_results_dir, f"Loss_{test_loss:.6f}.pth")
        torch.save({"predictions": all_preds, "targets": all_targets, "domain_size": all_domain_size}, results_path)
        self.LOG.info(f"Test loss: {test_loss:.6f}")
        self.LOG.info(f"Saved test results to {results_path}")
        return test_loss


class CorrectionTrainer(BaseTrainer1D):
    """
    Trainer for correction-based models. Uses simulation with correction terms.
    
    Inherits all parameters from BaseTrainer.
    """
    def __init__(self, optimizer, scheduler, train_param, writer, LOG=None, run_dir=None):
        super().__init__(optimizer, scheduler, train_param, writer, LOG, run_dir)
    
    def _log_outputs(self, correction, u, u_target, step, mode):
        """
        Log correction model outputs.
        
        Parameters:
            correction: Model's correction term
            u: Simulated states with correction
            u_target: Target trajectory
            step: Current global step
            mode: Training mode ('Train', 'Valid', 'Test')
        """
        if correction is None or self.LOG is None:
            return
        correction = correction.detach().cpu()
        u_target = u_target.detach().cpu()
        u = u.detach().cpu().numpy()
        
        # Only log if the correction term changes significantly
        if mode == "train" and (self.prev_NN_out is None or abs(correction.norm().item() - self.prev_NN_out.norm().item()) > 0.5 * self.prev_NN_out.norm().item()):
            self.prev_NN_out = correction
            self.LOG.info(f"{mode} correction - Mean: {self.prev_NN_out.mean().item():.4f}, Norm: {self.prev_NN_out.norm().item():.4f}")

    def process_batch(self, batch, steps, global_steps=None, mode=None):
        """
        Process a batch using simulation with correction approach.
        
        Parameters:
            batch: Dictionary containing simulation parameters and initial conditions
            steps: Number of simulation steps
            global_steps: Global step counter for logging
            mode: Training mode ('Train', 'Valid', 'Test')
            
        Returns:
            loss: Computed loss between simulated and target trajectories
            u_pred: Simulated trajectory with corrections (detached from computation graph)
            u_target: Target trajectory (moved to CPU)
            domain_size: Domain size (moved to CPU) (for post-process with physical parameter)
        """
        solver, u_initial, u_target, domain_size = self.train_param.init_sim_fn(batch, self.train_param)
        u_pred, correction = simulate_rollout(self.model, solver, u_initial, steps, self.train_param.correction_term, domain_size)  # [B,T,D]
        if correction is not None:
            self._log_outputs(correction, u_pred, u_target, global_steps, mode)
        # Add gradient penalty to loss
        if mode == "Train":
            gp_loss = self.train_param.lambda_gp * gradient_penalty(self.model, u_initial, domain_size) if self.train_param.lambda_gp > 0 else 0
            spectral_loss = self.train_param.lambda_spectrum * spectral_norm_regularization(self.model) if self.train_param.lambda_spectrum > 0 else 0
            loss = self.loss_fn(u_pred, u_target) + gp_loss + spectral_loss
        else:
            loss = self.loss_fn(u_pred, u_target)
        domain_size = domain_size.detach().cpu() if domain_size is not None else None
        return loss, u_pred.detach().cpu(), u_target.detach().cpu(), domain_size


class PredictionTrainer(BaseTrainer1D):
    """
    Trainer for prediction-based models. Uses direct rollout prediction.
    
    Inherits all parameters from BaseTrainer.
    """
    def __init__(self, optimizer, scheduler, train_param, writer, LOG=None, run_dir=None):
        super().__init__(optimizer, scheduler, train_param, writer, LOG, run_dir)
    
    def _log_outputs(self, prediction, u_target, step, mode):
        """
        Log prediction outputs.
        
        Parameters:
            prediction: Model's predicted trajectory (tensor)
            u_target: Target trajectory
            step: Current global step
            mode: Training mode ('Train', 'Valid', 'Test')
        """
        if prediction is None or self.LOG is None:
            return
        prediction = prediction.detach().clone().cpu()
        u_target = u_target.detach().clone().cpu()
        
        # Only log if the correction term changes significantly
        if mode == "Train" and (self.prev_NN_out is None or torch.norm(prediction - self.prev_NN_out).item() > 0.5 * torch.norm(self.prev_NN_out).item()):
            self.prev_NN_out = prediction
            self.LOG.info(f"{mode} prediction - Mean: {self.prev_NN_out.mean().item():.4f}, Norm: {self.prev_NN_out.norm().item():.4f}")

    def process_batch(self, batch, steps, global_steps=None, mode=None):
        """
        Process a batch using direct prediction approach.
        
        Parameters:
            batch: Tuple containing (initial_conditions, target_trajectory)
            steps: Number of rollout steps to predict
            global_steps: Global step counter for logging
            mode: Training mode ('Train', 'Valid', 'Test')
            
        Returns:
            loss: Computed loss between predictions and targets
            u_pred: Predicted trajectory (detached from computation graph) 
            u_target: Target trajectory (moved to CPU)
        """
        u_initial = batch[0].to(self.train_param.device)
        u_target = batch[1].to(self.train_param.device)
        ratio = self.train_param.dt / self.train_param.metadata["gen_dt"] 
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"dt must be a multiple of gen_dt ({self.train_param.metadata['gen_dt']})")
        else:
            ratio = int(round(ratio))
        domain_size = batch[2].to(self.train_param.device) if self.train_param.task == "KS" else None
        u_pred = predict_rollout(self.model, u_initial, steps, L=domain_size)  # [B,T,D]
        return self.loss_fn(u_pred, u_target), u_pred.detach().cpu(), u_target.detach().cpu(), domain_size
