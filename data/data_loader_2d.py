import glob,re,os,random
import torch
from data.DomainManager import DomainData
import lib.util.domain_io as domain_io
from torch.utils.data import Dataset, Subset
cuda_device = torch.device("cuda")
random.seed(42)
class GeoGroup:
    def __init__(self, geo_info, samples):
        """
        geo_info: tuple representing the geo-info
        samples: List of DomainData objects
        this class is for generating the dataset
        """
        self.geo_info = geo_info
        # self.samples = sorted(samples, key=lambda x: x.time_stamp)  
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DomainDataset(Dataset):
    def __init__(self, geo_groups, mstep=1, fixed_key=None):
        """
        geo_groups: Dictionary of GeoGroup objects
        mstep: Number of forward steps for the sequence
        fixed_key: If specified, always use this geo-info key (e.g., for testing)
        """
        self.geo_groups = geo_groups
        self.mstep = mstep
        self.fixed_key = fixed_key

        # Calculate valid lengths for each group
        self.valid_lengths = {key: len(group) - mstep for key, group in geo_groups.items()}
        if any(length < 0 for length in self.valid_lengths.values()):
            raise ValueError(f"All sequences must have at least {mstep} samples.")
        
        # Flatten all groups into a single indexable dataset
        self.flattened_index = []
        for key, length in self.valid_lengths.items():
            self.flattened_index.extend([(key, idx) for idx in range(length)])

    def __len__(self):
        # Total number of valid samples across all groups
        return len(self.flattened_index)

    def __getitem__(self, index):
        if self.fixed_key:
            # Select from the fixed key group if specified
            group_idx = index % self.valid_lengths[self.fixed_key]
            return self.get_sample(group_idx, self.fixed_key)

        # Retrieve the corresponding (config_key, group_idx) from the flattened index
        config_key, group_idx = self.flattened_index[index]
        return self.get_sample(group_idx, config_key)

    def get_sample(self, idx, config_key):
        geo_group = self.geo_groups[config_key]
        x = geo_group[idx].domain
        y = [geo_group[idx + i + 1].domain for i in range(self.mstep)]
        time_stamps = {
            "x": geo_group[idx].time_stamp,
            "y": [geo_group[idx + i + 1].time_stamp for i in range(self.mstep)]
        }
        return config_key, x, y, time_stamps


def collate_fn(batch):
    config_keys, x_batch, y_batch, time_stamps_batch = zip(*batch)
    return {
        'config_keys': config_keys,
        'x': x_batch,
        'y': y_batch,
        'time_stamps': time_stamps_batch
    }


def load_geo_groups(split, dtype, device, time_range=(0, None), downsample_factor=1, task=None):
    """
    Load geo-groups with a specified time range by indexing paths directly.
    
    Parameters:
        split (str): Dataset split identifier ("train", "valid", or "test").
        downsample_factor (int): Used for indicates the dataset.
        dtype: Data type for loading the domain.
        time_range (tuple): A tuple of (start_timestamp, end_timestamp) to slice paths directly, but unlike python slicing, the end_timestamp is inclusive.
        device: Device to load data onto.
        task: Task name (e.g., "BFS").

    Returns:
        dict: Dictionary of GeoGroup objects.
    """
    geo_groups = {}
    
    if downsample_factor == 4:
        # New pattern for the updated directory structure
        pattern = f"./INC_Data/{task}/Dataset/{split}/*/DownBFSdomain_*.json" 


    else:
        raise ValueError(f"No resolution for {downsample_factor}")
    
    paths = glob.glob(pattern)
    if paths == []:
        raise ValueError(f"No paths found with {pattern}.")
    
    paths.sort()
    start_timestamp, end_timestamp = time_range
    paths = [path[:-5] for path in paths]  # Remove '.json' extension
    
    base_step_pattern = r"Re([0-9]+)_Base([0-9\-]+)_Step([-+]?[0-9]*\.?[0-9]+)"
    # Updated pattern for new naming convention
    id_pattern = r"DownBFSdomain_(\d+)$"
    
    # Combined loop: filter by timestamp and load domains in one pass
    for path in paths:
        id_match = re.search(id_pattern, path)
        if not id_match:
            continue
            
        time_stamp = int(id_match.group(1))
        # Check if the timestamp falls within the specified range
        if not ((start_timestamp is None or time_stamp >= start_timestamp) and 
                (end_timestamp is None or time_stamp <= end_timestamp)):
            continue
            
        match = re.search(base_step_pattern, path)
        if match and id_match:
            Re = int(match.group(1))
            base_str = match.group(2)
            s = float(match.group(3))
            base = list(map(int, base_str.split("-")))
            geo_info = (tuple(base), s, Re)
            domain = domain_io.load_domain(path, dtype=dtype, device=device)
            domain_data = DomainData(
                domain=domain,
                time_stamp=time_stamp
            )
            if geo_info not in geo_groups:
                geo_groups[geo_info] = GeoGroup(geo_info, [])
            geo_groups[geo_info].samples.append(domain_data)
        else:
            raise ValueError(f"Invalid path format: {path}")
    
    return geo_groups


def get_stats(geo_groups, task=None):
    u_values = []
    v_values = []
    nu_values = []
    for item in geo_groups.values():
        for domain_data in item.samples:
            nu_values.append(domain_data.domain.viscosity)
            for block in domain_data.domain.getBlocks():
                u_values.append(block.velocity[0, 0])
                v_values.append(block.velocity[0, 1])
    u_flat=torch.cat([values.flatten() for values in u_values])
    v_flat=torch.cat([values.flatten() for values in v_values])
    nu_flat=torch.cat([values.flatten() for values in nu_values])
    if task == "Karman":
        Re_norm = 1000
        geo_norm = 2
    elif task == "BFS":
        Re_norm = 1400
        geo_norm = 1
    else:
        raise ValueError("Invalid task.")
    return {"u_max":u_flat.max(),"u_min":u_flat.min(),"v_max":v_flat.max(),"v_min":v_flat.min(), "u_mean":u_flat.mean(), "v_mean":v_flat.mean(), "u_std":u_flat.std(), "v_std":v_flat.std(), "Re_norm":Re_norm, "geo_norm":geo_norm, "nu_max":nu_flat.max(), "nu_min":nu_flat.min(), "nu_mean":nu_flat.mean(), "nu_std":nu_flat.std()}


    
def load_data(split, dtype, task, run_dir, time_range=(0, None), downsample_factor=1, mstep=5, shuffle=False, data_norm=None, num_batches=None):
    """
    Load data into a DataLoader with optional random sampling of batches.

    Parameters:
        split (str): Dataset split identifier ("train", "valid", or "test").
        dtype: Data type.
        task: Task name (e.g., "Karman" or "BFS").
        run_dir: Directory to store output statistics.
        time_range: Range of timestamps to filter data.
        downsample_factor: Data resolution factor.
        mstep: Number of forward steps in each sequence.
        shuffle: Whether to shuffle the DataLoader.
        data_norm: If True, compute and save dataset statistics.
        num_batches: If specified, sample this many batches.
    """
    geo_groups = load_geo_groups(split, dtype=dtype, device=cuda_device, time_range=time_range, downsample_factor=downsample_factor, task=task)
    dataset = DomainDataset(geo_groups, mstep=mstep)
    # for validation, using limited number of samples
    if num_batches:
        num_valid_samples = len(dataset)
        if num_batches > num_valid_samples:
            raise ValueError(f"Requested {num_batches} batches, but only {num_valid_samples} valid samples are available.")
        # selected_indices = random.sample(range(num_valid_samples), num_batches)
        selected_indices = list(range(0, num_valid_samples, num_valid_samples//num_batches))[:num_batches]
        dataset = Subset(dataset, selected_indices)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=shuffle)
    stats = None
    if data_norm is not None:
        stats = get_stats(geo_groups, task)
        output_dir = os.path.join(run_dir, "Train_stats")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(stats, f"{output_dir}/stats.pkl")
    return data_loader, stats