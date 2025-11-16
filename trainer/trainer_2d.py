import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, glob, re, gc
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
import PISOtorch # core CUDA module
from solvers.solver_2d import PISOtorch_diff # differentiable wrapper for PISO
from solvers.solver_2d import PISOtorch_simulation # helper for PISO loop
from lib.util.logging import setup_run, get_logger, close_logging # logging and output
from data.data_loader_2d import load_data
import pandas as pd
from trainer.Loss import LossAccumulator, ttonp


class Trainer:
    def __init__(self, sim_params,train_params,run_dir, model_dir=None, LOG=None, task=None):
        # herein, this domain_manager is used initialize all the thing required for simulation, including domain, prep_fn, layout. But the domain is only for init CNN, where the dimension, and block connection is needed. The geo information isn't needed, so we can use any of the geo_info. The prep_fn is a template function for simulator, and at this stage it only consisit the advection boundary function. 
        self.sim_params = sim_params
        self.train_params = train_params
        self.method = train_params.method
        self.run_dir = run_dir
        self.task = task
        self.model_dir = model_dir
        self.checkID = None
        self.start_points = None
        self.Corrector = self.train_params.Corrector
        self.prep_fn_key = self.train_params.prep_fn_key
        self.LOG = LOG
        self._setup_domain_manager()
        # Initialize training components
        self._initialize_components()
        self.LOG.info(f"DomainManager parameters: {self.sim_params.__dict__}, Train parameters: {self.train_params.__dict__}")
    
    def _setup_domain_manager(self):
        if self.task == "Karman":
            from data.DomainManager import KarmanDomainManager
            self.domain_manager = KarmanDomainManager(**self.sim_params.__dict__)
        elif self.task == "BFS":
            from data.DomainManager import BFSDomainManager
            self.domain_manager = BFSDomainManager(**self.sim_params.__dict__)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        self.config_keys = self.domain_manager.get_all_keys()[0]
    
    def _initialize_components(self):
        # Initialize domain manager, model(corrector), optimizer, and scheduler
        self.sim_params.layout = self.domain_manager.get_config(self.config_keys).layout
        self.corrector = self.Corrector(domain=self.domain_manager.get_config(self.config_keys).domain, LOG=self.LOG,train_params=self.train_params, sim_params=self.sim_params)
        self.prep_fn = self.domain_manager.get_config(self.config_keys).prep_fn
        self.optimizer = optim.Adam(self.corrector.parameters(), lr=self.train_params.lr, weight_decay=self.train_params.weight_decay)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2, verbose=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2)

        self._clean_up(self.domain_manager.get_config(self.config_keys).domain)# Delete placeholder domain
        # Simulation setup
        self.solver = PISOtorch_simulation.Simulation(
            domain=None, time_step=self.sim_params.time_step, block_layout=self.sim_params.layout ,
            prep_fn=self.prep_fn, substeps=self.sim_params.substeps, corrector_steps=2,
            pressure_tol=self.sim_params.pressure_tol, advect_non_ortho_steps=1, differentiable=True,pressure_non_ortho_steps=1, pressure_return_best_result=True, velocity_corrector="FD",non_orthogonal=False, norm_vel=True, log_dir=os.path.join(self.run_dir, "train_snippet"), log_interval=1, save_domain_name=None, stop_fn=None, adaptive_CFL=self.sim_params.adaptive_CFL)
        # key solver, which wraps the PISO module for simulation loop        
    def _prepare_batch(self, batch):
        domain = batch['x'][0].Clone()
        target_domain = batch['y'][0]
        # Set Re, geo_feature for corrector based on batch
        self.corrector.Re = torch.tensor(batch['config_keys'][0][2], device=cuda_device)
        self.corrector.geo_feature = torch.tensor(batch['config_keys'][0][1], device=cuda_device)
        return domain, target_domain
    

    def _append_corrector_to_prep_fn(self):
        if (self.prep_fn_key not in self.prep_fn or self.corrector.correct_prediction not in self.prep_fn[self.prep_fn_key]):
            PISOtorch_simulation.append_prep_fn(self.prep_fn, self.prep_fn_key, self.corrector.correct_prediction)
    
    def _create_loss_acc(self, target_domain,stats=None):
        return LossAccumulator(
            target_domain,layout=self.sim_params.layout ,dtype=self.train_params.dtype, skip_blocks=self.train_params.skip_blocks, LOG=self.LOG, stats=stats)
    
    def _simulate(self, domain, loss_acc, mstep, log_images=False):
        domain.PrepareSolve()
        self.solver.domain, self.solver.log_fn, self.solver.log_images = domain, loss_acc.add_loss, log_images
        self.solver.make_divergence_free()
        self._append_corrector_to_prep_fn()
        self.solver.run(mstep,log_domain=False)
        loss = loss_acc.get_loss_normalized()
        if loss.isnan().any():
            raise RuntimeError("loss is NaN.")
        return loss
    
    def _clean_up(self, domain):
        domain.Detach()
        del domain
        gc.collect()
        torch.cuda.empty_cache()

    def _process_batch(self, batch, mode="train", start_point=None):
        mode_config = {
        "train": {"mstep": self.train_params.mstep, "requires_grad": True, "log_images": False, "stats": self.stats if self.train_params.scale_loss else None},
        "validate": {"mstep": self.train_params.valid_steps, "requires_grad": False, "log_images": False, "stats": None},
        "test": {"mstep": self.train_params.test_steps, "requires_grad": False, "log_images": False, "stats": None},
        }
        if mode not in mode_config:
            raise ValueError(f"Unknown mode: {mode}")
        config = mode_config[mode]
        mstep, requires_grad, log_images, stats = config["mstep"], config["requires_grad"], config.get("log_images"), config.get("stats")
        with torch.set_grad_enabled(requires_grad):
            domain, target_domain = self._prepare_batch(batch)
            loss_acc = self._create_loss_acc(target_domain, stats=stats)
            loss_acc.reset_loss()
            if mode == "test" and start_point is not None:
                self.solver.img_out_idx = start_point
                self.solver.log_images = log_images
            self.optimizer.zero_grad() if requires_grad else None
            loss = self._simulate(domain, loss_acc, mstep, log_images=log_images)
            
            if requires_grad:
                loss.backward()
                self.optimizer.step()
            
            final_loss_val = loss.item()
            results = loss_acc.results
            correction = self.corrector.correction
            
            self.corrector.init_correction()
            self._clean_up(domain)
            return final_loss_val, results, correction
 
    def _train_batch(self, batch, epoch, batch_idx, num_batches):
        train_loss, results, correction = self._process_batch(batch, mode="train")
        self.LOG.info(f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | Train Loss: {train_loss:.6f}")
        return train_loss, results, correction
    
    def _validate_batch(self, batch):
        val_loss, results, correction = self._process_batch(batch, mode="validate")
        return val_loss, results, correction
    
    def _test_batch(self, batch, start_point):
        test_loss, results, correction = self._process_batch(batch, mode="test", start_point=start_point)
        return test_loss, results, correction

    def _initialize_training(self):
        train_loader, self.stats = load_data(
            split="train", run_dir=self.run_dir, dtype=self.sim_params.dtype,
            task=self.task, time_range=self.train_params.train_range, downsample_factor=4,
            mstep=self.train_params.mstep, data_norm=self.train_params.data_norm, shuffle=True
        )
        val_loader, _ = load_data(
            split="valid", run_dir=self.run_dir, dtype=self.sim_params.dtype,
            task=self.task, time_range=self.train_params.valid_range, downsample_factor=4,
            mstep=self.train_params.valid_steps, num_batches=5, shuffle=False
        )
        self.corrector.stats = self.stats
        self.LOG.info(f"Train with Stats: {self.stats}") if self.stats is not None else self.LOG.info("Train without Stats.")
        self.LOG.info(f"--- Training loop started with mstep: {self.train_params.mstep} with method {self.method} ---")
        self._load_model(self.checkID)
        return train_loader, val_loader
    
    def _initialize_testing(self, checkID=None):
        if self.model_dir is None:
            raise ValueError("No model directory provided for testing.")
        best_model_path=self._load_model(checkID)
        model_name = os.path.splitext(os.path.basename(best_model_path))[0]
        output_dir = os.path.join(self.run_dir, "test_results")
        os.makedirs(output_dir, exist_ok=True)
        if self.start_points is None:
            raise ValueError("No start points provided for testing.")
        self.stats=torch.load(os.path.join(self.run_dir, "Train_stats/stats.pkl"))
        self.corrector.stats=self.stats
        self.LOG.info(f"Test with Stats: {self.stats}") if self.stats is not None else self.LOG.info("Test without Stats.")
        return output_dir, model_name
    
    def _lr_step(self, val_loss):
        if val_loss < float('inf'):
            self.scheduler.step(val_loss)
            self.LOG.info(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']} due to plateau.")
        else:
            self.LOG.warning(f"Validation failed. Skipping scheduler step.")

    def _save_checkpoint(self, epoch, epoch_train_loss, val_loss=None, best_val_loss=float('inf'),best_model_path=None, patience_counter=0):
        os.makedirs(os.path.join(self.run_dir, "models/check"), exist_ok=True)
        checkpoint_path = os.path.join(self.run_dir, f"models/check/checkpoint_{epoch}_{epoch_train_loss: .2f}".replace('.', '_')+".pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.corrector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": epoch_train_loss,
            "val_loss": val_loss,
        }, checkpoint_path)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(self.run_dir, f"models/best_valid_model_epoch{epoch}_{val_loss:.2f}".replace('.', '_') + ".pth")
            torch.save(self.corrector.state_dict(), best_model_path)
        else:
            patience_counter += 1
            self.LOG.info(f"EarlyStopping counter: {patience_counter} out of {self.train_params.early_stop_patience}")
        return best_val_loss, best_model_path, patience_counter

    def train(self):
        # stats depends on self.train_params.data_norm, if None, stats=None
        train_loader, val_loader = self._initialize_training()
        best_val_loss, best_model_path, patience_counter = float('inf'), None, 0
        num_batches = len(train_loader)
        for epoch in range(self.train_params.epochs):
            self.LOG.info(f"--- Epoch {epoch} ---")
            self.corrector.train()
            epoch_train_loss = 0
            epoch_results = []
            for batch_idx, batch in enumerate(train_loader):
                batch_train_loss, details, correction = self._train_batch(batch, epoch, batch_idx, num_batches)
                epoch_train_loss += batch_train_loss
                epoch_results.append({"Epoch": epoch,"Index": batch_idx, "data": details, "correction": correction})
            epoch_train_loss /= num_batches
            self.LOG.info(f"Epoch {epoch} train loss: {epoch_train_loss}")

            # Validation step and saving best model
            val_loss = self.validate(val_loader, epoch)
            self.LOG.info(f"End of Epoch {epoch} | Training Loss: {epoch_train_loss}, Validation Loss: {val_loss}")
            self._lr_step(val_loss)
            #chekpoint saving
            best_val_loss, best_model_path, patience_counter=self._save_checkpoint(epoch, epoch_train_loss, val_loss, best_val_loss, best_model_path, patience_counter)
            if patience_counter >= self.train_params.early_stop_patience:
                self.LOG.info(f"Early stopping at epoch {epoch}.")
                break


    def validate(self, val_loader,epoch):
        self.corrector.eval()
        val_loss_sum = 0
        if len(val_loader) == 0:
            self.LOG.error("Validation loader is empty.")
            return float('inf')
        with torch.no_grad():
            for val_idx,batch in enumerate(val_loader):
                try:
                    loss_val, details, correction = self._validate_batch(batch)
                    val_loss_sum += loss_val
                except Exception as e:
                    self.LOG.error(f"Validation failed: {e} at {val_idx}")
                    return float('inf')
        avg_val_loss = val_loss_sum / len(val_loader)
        self.LOG.info(f"Validation loss: {avg_val_loss}")
        return avg_val_loss

    def test(self,checkID=None):
        output_dir, model_name = self._initialize_testing(checkID)
        for start_point in self.start_points:
            results = []
            self.LOG.info(f"Testing step started with methods and model: {self.method}, {model_name}, at starting point{start_point}")
            test_loader, _ = load_data(split="test", time_range=(start_point, start_point+self.train_params.test_steps), downsample_factor=4, mstep=self.train_params.test_steps, shuffle=False, dtype=self.sim_params.dtype, task=self.task, run_dir=self.run_dir)
            test_loss = 0
            with torch.no_grad():
                for test_idx, batch in enumerate(test_loader):
                    #since when we init the config_keys, we used the geo for test, so it need to be the same as the test data
                    if batch['config_keys'][0] != self.config_keys:
                        raise ValueError(f"Config keys do not match: {batch['config_keys']} vs {self.config_keys}")
                    try:
                        loss_batch, details, correction = self._test_batch(batch, start_point)
                        test_loss += loss_batch
                        failure = ""
                    except Exception as e:
                        self.LOG.error(f"Testing failed at batch {test_idx}: {e}")
                        failure = "(Failed)"
                        pass
                    results.append({"Index": test_idx, "data": details, "correction": correction})
            pd.DataFrame(results).to_pickle(os.path.join(output_dir, f"test_results_{model_name}_Start{start_point}_Forward{self.train_params.test_steps}{failure}.pkl"))
    
    def _get_check_id(self,model_dir):
        checkpoint_files = glob.glob(os.path.join(model_dir, "models/check/checkpoint_*.pth"))
        if not checkpoint_files:
            return None
        epoch_numbers = []
        for file in checkpoint_files:
            match = re.search(r"checkpoint_(\d+)_", file)  # Extract epoch number from filename
            if match:
                epoch_numbers.append(int(match.group(1)))
        if not epoch_numbers:
            return None
        return max(epoch_numbers)

    def _load_model(self,checkID):
        best_model_path = None
        if self.model_dir is not None:
            best_model_path = self.get_model_dir(self.model_dir,checkID)[0]
            state_dict = torch.load(best_model_path)
            if "model_state_dict" in state_dict:
                self.corrector.load_state_dict(state_dict["model_state_dict"])
                self.LOG.info(f"Loaded model from {best_model_path} with checker point for training.")
            else:
                self.corrector.load_state_dict(state_dict)
                self.LOG.info(f"Loaded model from {best_model_path} for training.")
        return best_model_path
    
    def get_model_dir(self, model_dir,checkID=None):
        if checkID is not None:
            model_paths = glob.glob(os.path.join(model_dir, f"models/check/checkpoint_{checkID}*.pth"))
        else:
            model_paths = glob.glob(os.path.join(model_dir, "models/best_valid_model*.pth"))
        model_paths.sort(key=lambda path: float(re.search(r"(\d+)", path).group()))
        if model_paths == []:
            checkID = self._get_check_id(model_dir) # load the last model
            self.LOG.warning(f"No model found in {model_dir}, path from checkpoint {checkID}")
            model_paths = glob.glob(os.path.join(model_dir, f"models/check/checkpoint_{checkID}*.pth"))
        return model_paths

    def _get_check_id(self,model_dir):
        checkpoint_files = glob.glob(os.path.join(model_dir, "models/check/checkpoint_*.pth"))
        if not checkpoint_files:
            return None
        epoch_numbers = []
        for file in checkpoint_files:
            match = re.search(r"checkpoint_(\d+)_", file)  # Extract epoch number from filename
            if match:
                epoch_numbers.append(int(match.group(1)))
        if not epoch_numbers:
            return None
        return max(epoch_numbers)

    def _inverted_exponential_wd(self):
        k = 5 / (self.train_params.epochs - 1)
        start_wd = self.train_params.weight_decay
        end_wd = self.train_params.weight_decay_end if hasattr(self.train_params, 'weight_decay_end') else 0.5
        weight_decay_list= [end_wd - (end_wd - start_wd) * torch.exp(torch.tensor(-k * epoch)) for epoch in range(self.train_params.epochs)]
        self.LOG.info(f"Weight decay schedule initialized: start={start_wd}, end={end_wd},result={weight_decay_list}")
        return weight_decay_list
    
    def _linear_wd(self):
        start_wd = self.train_params.weight_decay
        end_wd = self.train_params.weight_decay_end if hasattr(self.train_params, 'weight_decay_end') else 0.5
        epochs = self.train_params.epochs
        
        # Generate the linear schedule
        weight_decay_list = [start_wd + (end_wd - start_wd) * epoch / (epochs - 1) for epoch in range(epochs)]
        
        self.LOG.info(f"Weight decay schedule initialized (linearly): start={start_wd}, end={end_wd}, result={weight_decay_list}")
        return weight_decay_list
    
    def _exponential_wd(self):
        start_wd = self.train_params.weight_decay
        end_wd = self.train_params.weight_decay_end if hasattr(self.train_params, 'weight_decay_end') else 0.5
        epochs = self.train_params.epochs
        
        # Exponential decay factor
        decay_factor = (end_wd / start_wd) ** (1 / (epochs - 1))
        
        # Generate the exponential schedule
        weight_decay_list = [start_wd * (decay_factor ** epoch) for epoch in range(epochs)]
        
        self.LOG.info(f"Exponential weight decay schedule initialized: start={start_wd}, end={end_wd}, result={weight_decay_list}")
        return weight_decay_list

    def _get_WD(self):
        if self.train_params.dynamic_weight_decay == "iexp":
            return self._inverted_exponential_wd()
        elif self.train_params.dynamic_weight_decay == "exp":
            return self._exponential_wd()
        elif self.train_params.dynamic_weight_decay == "linear":
            return self._linear_wd()
        else:
            raise ValueError(f"Unknown weight decay schedule: {self.train_params.dynamic_weight_decay}")