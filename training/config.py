from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


LOCAL_DIR = Path(__file__).resolve().parent / "_local"


@dataclass
class Settings:
    seed: int = 42
    split_seed: int = 42
    start_tolerance_m: float = 0.25
    shared_conditions: tuple = tuple(f"{time}_{weather}" for time in ("day", "sunset", "night")
                                     for weather in ("sunny", "cloudy", "rainy"))
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    spawn_round_decimals: int = 2
    safety_buffer_m: float = 1.0
    max_gap_m: float = 9.0
    switching_threshold: float = 0.0
    visual_history_frames: int = 1  # Legacy checkpoint default; new CLI runs use five.
    history_steps: int = 5
    horizon_steps: int = 10
    boundary_band_m: float = 0.25

    # For pose i, the action row is i + action_offset and gap row i + gap_offset.
    # These are the archive audit's best-supported offsets, not capture timestamps.
    action_offset: int = 0
    gap_offset: int = -1
    dt: float = 0.1
    longitudinal_saturation: float = 3.0
    action_low: tuple = (-3.0, -1.0)
    action_high: tuple = (3.0, 1.0)

    hidden_sizes: tuple = (128, 128)
    batch_size: int = 256
    risk_lr: float = 0.001
    risk_epochs: int = 40
    patience: int = 5
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    updates: int = 10_000
    dataset_selection: str = "combined"


def setup_device(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training device: CUDA ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        device = torch.device("cpu")
        print("Training device: CPU", flush=True)
    return device


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("Unexpected numerical cache format. Run training.prepare_data.")
    return data
