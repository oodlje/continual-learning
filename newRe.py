from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm


import sys
sys.path.append('/home/jieun/project/content/l5kit/l5kit')

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace, rmse, _average_displacement_error, _final_displacement_error, detect_collision, distance_to_reference_trajectory
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os
import time
import tracemalloc
import torch


torch.cuda.is_available()


os.environ["L5KIT_DATA_FOLDER"] = "/home/jieun/project/content/l5kit/examples/agent_motion_prediction/prediction-dataset"
dm = LocalDataManager(None)

cfg = load_config_data("/home/jieun/project/content/l5kit/examples/agent_motion_prediction/agent_motion_config.yaml")
print(cfg)


#################################Added this to include other models##########################################

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_model(cfg: Dict) -> torch.nn.Module:
    # load pretrained weights via the new API
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # --- 1) replace the initial conv ---
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels

    first_conv: nn.Conv2d = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=num_in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False
    )

    # --- 2) replace the classifiers Linear layer ---
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    old_fc: nn.Linear = model.classifier[1]
    model.classifier[1] = nn.Linear(
        in_features=old_fc.in_features,
        out_features=num_targets
    )

    return model


def forward(data, model, device, criterion):
    # 1) Pull everything onto the device
    images = data["image"]
    targets = data["target_positions"]
    masks   = data["target_availabilities"]

    # If they came in as NumPy arrays, convert once
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    images = images.to(device)       # --> [B,C,H,W] or [C,H,W]
    targets = targets.float().to(device)  # --> [B,F,2] or [F,2]
    masks = masks.float().to(device)      # --> [B,F]   or [F]

    # 2) Make sure theres always a batch dimension
    if images.dim() == 3:             # single sample: [C,H,W]
        images = images.unsqueeze(0)  # --> [1,C,H,W]
    if targets.dim() == 2:            # single sample: [F,2]
        targets = targets.unsqueeze(0)     # --> [1,F,2]
    if masks.dim() == 1:              # single sample: [F]
        masks = masks.unsqueeze(0)         # --> [1,F]

    # 3) Now masks is [B,F], we want [B,F,1] to broadcast over the 2 coords
    masks = masks.unsqueeze(-1)       # --> [B,F,1]

    # 4) Forward
    outputs = model(images)           # --> [B, 2*F]
    outputs = outputs.view_as(targets)  # --> [B,F,2]

    # 5) Compute loss
    loss = criterion(outputs, targets)  # --> [B,F,2]
    loss = (loss * masks).mean()        # zero out invalid timesteps
    return loss, outputs

#########################################################################################################

import os
import torch
from torch.utils.data import DataLoader, Subset

# Ensure compatibility with numpy updates
import numpy as np
np.int = np.int64
np.bool = np.bool_

# Configuration and rasterizer setup
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)

# ===== LOAD FULL DATASET =====
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# Dynamically set number of workers based on available CPU cores
#max_workers = os.cpu_count()  # Get the number of CPU cores
#train_cfg["num_workers"] = min(train_cfg.get("num_workers", 0), max_workers // 2)

# ===== CREATE DATALOADER =====
train_dataloader = DataLoader(
    train_dataset,
    #shuffle=True,  # Shuffle within the subset for training
    shuffle=train_cfg["shuffle"],
    batch_size=train_cfg["batch_size"],
    num_workers=train_cfg["num_workers"]
)

# ===== DEBUG =====
print(f"Total dataset size: {len(train_dataset)}")
print(f"Number of DataLoader workers: {train_cfg['num_workers']}")


#print(f"Example batch: {next(iter(train_dataloader))}")
print(train_dataset)

# ==== INIT MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="none")

import tracemalloc
import numpy as np
from tqdm import tqdm
import torch
import psutil

# Initialize variables to track memory
process = psutil.Process()  # Get the current process
peak_ram_mb = 0  # Peak RAM usage in MB


# Initialize tracemalloc for Python memory tracking
tracemalloc.start()

print(np.__version__)
np.int = np.int64
np.bool = np.bool_
start_all = time.time()

print("\n***************************CL begins here with Mobilenet********************************")

# =========================
# Create Tasks by Percentile

# =========================
from torch.utils.data import DataLoader, random_split, Subset
import torch
import numpy as np
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Step 1: Randomly sample 10% of the whole dataset for testing EWC
dataset_size = len(train_dataset)
sample_size = int(0.2 * dataset_size)

# Randomly sample 10%
#task_data, _ = random_split(train_dataset, [sample_size, dataset_size - sample_size])

# Optional: check structure of one sample
#sample_data = next(iter(DataLoader(task_data, batch_size=1)))
#print("Sample keys:", sample_data.keys())

indices = torch.randperm(dataset_size)[:sample_size]
sampled_subset = Subset(train_dataset, indices)

# Optional: check structure of one sample
sample_loader = DataLoader(sampled_subset, batch_size=1)
sample_data = next(iter(sample_loader))
print("Sample keys:", sample_data.keys())

# Step 2: Materialize the sampled data into a list (actual samples)
sampled_list = [train_dataset[i] for i in indices]
print("Sampled dataset size:", len(sampled_list))


def create_tasks_by_velocity_percentile(dataset, num_tasks=3):
    """
    Splits the dataset into `num_tasks` subsets with equal sample sizes
    based on the sorted target velocity magnitudes.
    """
    # Compute the norm (magnitude) of the first future velocity for all samples
    all_speeds = [torch.norm(torch.tensor(sample["target_velocities"][0])).item() for sample in dataset]

    # Sort indices by speed magnitude
    sorted_indices = np.argsort(all_speeds)
    sorted_speeds = np.array(all_speeds)[sorted_indices]
    total_samples = len(sorted_indices)
    samples_per_task = total_samples // num_tasks

    # Create equal-sized tasks
    tasks = []
    for i in range(num_tasks):
        start_idx = i * samples_per_task
        end_idx = (i + 1) * samples_per_task if i < num_tasks - 1 else total_samples
        indices = sorted_indices[start_idx:end_idx].tolist()
        #tasks.append(Subset(dataset, indices))
        task_subset = [dataset[j] for j in indices]
        tasks.append(task_subset)

# Print range of velocities in this task
        task_speeds = sorted_speeds[start_idx:end_idx]
        min_speed = float(np.min(task_speeds))
        max_speed = float(np.max(task_speeds))
        print(f"  - Task {i + 1}: {len(task_subset)} samples | Velocity range: {min_speed:.2f} ~ {max_speed:.2f} m/s")

    return tasks
import time
start_it = time.time()

# Create and display task info
#tasks = create_tasks_by_velocity_percentile(task_data, num_tasks=3)
tasks = create_tasks_by_velocity_percentile(sampled_list, num_tasks=3)

print("\nCreated Tasks Based on Velocity Percentiles:")
for idx, task in enumerate(tasks):
    print(f"  - Task {idx + 1}: {len(task)} samples")

end_it = time.time()
task_n_time = end_it - start_it
print(f"Creating the tasks is completed in {task_n_time:.2f} seconds\n")





print("train the rehearsal Mobilenet model task by task")

import tracemalloc
import psutil
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import time

# Rehearsal memory for replay
rehearsal_memory = []
max_memory_size = 3000  # Define the maximum memory size

def rehearsal_update(memory, new_data, device):
    """
    Add each sample in new_data to rehearsal memory.
    new_data may be a single-sample dict, or a batch dict.
    Ensure we always iterate over the batch dimension.
    """
    # extract
    imgs   = new_data["image"]
    labs   = new_data["target_positions"]
    # if single-sample, add a batch dim
    if imgs.dim() == 3:           # [C, H, W]
        imgs = imgs.unsqueeze(0)  # --> [1, C, H, W]
        labs = labs.unsqueeze(0)  # --> [1, T, 2]

    # move everything to the device once
    imgs = imgs.to(device)
    labs = labs.to(device)

    # now imgs: [B, C, H, W], labs: [B, T, 2]
    for img, lab in zip(imgs, labs):
        memory.append((img, lab))

    # keep only the last max_memory_size items
    if len(memory) > max_memory_size:
        del memory[:-max_memory_size]

    return memory

def get_replay_batch(memory, batch_size, device, expected_num_frames=50):
    """Sample a batch from rehearsal memory."""
    if len(memory) == 0:
        return None

    sample_size = min(batch_size, len(memory))
    sampled = random.sample(memory, sample_size)

    # Each item in memory is (image_tensor, target_tensor)
    images = torch.stack([img for img, _ in sampled]).to(device)    # --> [B,C,H,W]
    targets = torch.stack([tgt for _, tgt in sampled]).to(device)  # --> [B,T,2]

    # 1) ensure batch dim
    if targets.dim() != 3:
        raise RuntimeError(f"Expected targets.ndim==3, got {targets.shape}")

    # 2) Pad or crop in time to exactly expected_num_frames
    B, T, _ = targets.shape
    if T < expected_num_frames:
        pad = torch.zeros(B, expected_num_frames - T, 2, device=device)
        targets = torch.cat([targets, pad], dim=1)
    elif T > expected_num_frames:
        targets = targets[:, :expected_num_frames, :]

    # 3) Always produce a mask of shape [B,expected_num_frames]
    avail = torch.ones(B, expected_num_frames, device=device)

    return {
        "image": images,                    # [B,C,H,W]
        "target_positions": targets,        # [B,F,2]
        "target_availabilities": avail      # [B,F]
    }



# Set up memory tracking
tracemalloc.start()
process = psutil.Process(os.getpid())

# Define replay batch size
replay_batch_size = 12

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()


for task_id, task_loader in enumerate(tasks):
    losses_train = [] 
    tr_it = iter(task_loader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

    peak_python_memory_mb = 0.0
    peak_ram_mb = 0.0

    for step in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(task_loader)
            data = next(tr_it)

        # Convert all numpy arrays in the batch to torch tensors
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])

        # Move current data to device (GPU)
        data["image"] = data["image"].to(device)
        data["target_positions"] = data["target_positions"].to(device)

        # Update rehearsal memory with current data, move to the same device as the model
        rehearsal_memory = rehearsal_update(rehearsal_memory, data, device)

        # Get a replay batch from the rehearsal memory and ensure it's moved to the device (GPU)
        replay_data = get_replay_batch(rehearsal_memory, replay_batch_size, device)

        # Train model on current task data
        model.train()
        torch.set_grad_enabled(True)

        # Compute loss for current data
        loss, _ = forward(data, model, device, criterion)

        # Rehearsal data loss (only compute if replay_data is not None)
        if replay_data:
            replay_loss, _ = forward(replay_data, model, device, criterion)
        else:
            replay_loss = torch.tensor(0.0).to(device=device)

        # Total loss (current loss + replay loss)
        total_loss = loss + replay_loss

        # Backward pass and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log loss and memory usage
        losses_train.append(total_loss.item())

        # --- Memory Monitoring ---
        current_py, peak_py = tracemalloc.get_traced_memory()
        peak_python_memory_mb = max(peak_python_memory_mb, peak_py / 1e6)

        current_ram_mb = process.memory_info().rss / 1e6
        peak_ram_mb = max(peak_ram_mb, current_ram_mb)

        # Update tqdm description with memory stats
        progress_bar.set_description(
            f"Task {task_id + 1} | Step {step + 1}/{cfg['train_params']['max_num_steps']} | "
            f"loss: {total_loss.item():.4f} | avg_loss: {np.mean(losses_train):.4f} | "
            f"peak_py_mem: {peak_python_memory_mb:.2f} MB | "
            f"current_ram: {current_ram_mb:.2f} MB | peak_ram: {peak_ram_mb:.2f} MB"
        )

    # End of task logging
    print(f"\nTraining on Task {task_id + 1} completed")
    print(f"Peak Python memory during Task {task_id + 1}: {peak_python_memory_mb:.2f} MB")
    print(f"Peak RAM usage during Task {task_id + 1}: {peak_ram_mb:.2f} MB\n")

end_time = time.time()
training_time = end_time - start_time
print(f"Training with Rehearsal completed in {training_time:.2f} seconds")

# ===== GENERATE AND LOAD CHOPPED DATASET FOR EWC
num_frames_to_chop = 100
eval_cfg = cfg["val_data_loader"]
eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"],
                              num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)


eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
eval_mask_path = str(Path(eval_base_path) / "mask.npz")
eval_gt_path = str(Path(eval_base_path) / "gt.csv")

eval_zarr = ChunkedDataset(eval_zarr_path).open()
eval_mask = np.load(eval_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)



# ==== EVAL LOOP
model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
agent_ids = []

progress_bar = tqdm(eval_dataloader)
for data in progress_bar:
    _, ouputs = forward(data, model, device, criterion)

    # convert agent coordinates into world offsets
    agents_coords = ouputs.cpu().numpy()
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()
    coords_offset = transform_points(agents_coords, world_from_agents) - centroids[:, None, :2]

    future_coords_offsets_pd.append(np.stack(coords_offset))
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

pred3_path = f"{gettempdir()}/pred_rehear.csv"

write_pred_csv(pred3_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
              )

metrics = compute_metrics_csv(eval_gt_path, pred3_path, [neg_multi_log_likelihood,rmse, time_displace])
for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)
