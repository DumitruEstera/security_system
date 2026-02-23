"""
Configuration for the Security Action Recognition System.
All hyperparameters, paths, and class definitions are centralized here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────
# Action classes that the SlowFast model will recognize
# ──────────────────────────────────────────────────────────────────────
ACTION_CLASSES = [
    "normal",               # 0 - No anomaly
    "vandalism",            # 1
    "harassment",           # 2
    "fighting",             # 3 - Physical altercation
    "dangerous_object",     # 4 - Handling dangerous objects
    "faint",                # 5 - Fainting / collapse / fall
    "suspicious_gathering", # 6
]

NUM_CLASSES = len(ACTION_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(ACTION_CLASSES)}


@dataclass
class DatasetConfig:
    """Paths and labels for each source dataset."""
    name: str
    root_dir: str                    # Root directory of the dataset
    label_map: Dict[str, int]        # Maps dataset-specific labels → our class indices
    video_ext: str = ".avi"          # Video file extension
    split_file: Optional[str] = None # Optional train/test split file


@dataclass
class ModelConfig:
    """SlowFast model configuration."""
    model_name: str = "slowfast_r50"
    pretrained: bool = True
    num_classes: int = NUM_CLASSES
    dropout_rate: float = 0.5
    freeze_backbone_epochs: int = 5  # Freeze backbone for first N epochs


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./output/checkpoints"
    log_dir: str = "./output/logs"

    # Video clip parameters
    clip_duration_sec: float = 2.0    # Duration of each clip in seconds
    num_frames_slow: int = 8          # Frames for the slow pathway
    num_frames_fast: int = 32         # Frames for the fast pathway (4× slow)
    sampling_rate_slow: int = 8       # Temporal stride for slow pathway
    sampling_rate_fast: int = 2       # Temporal stride for fast pathway
    crop_size: int = 224              # Spatial crop size

    # Training
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 3
    lr_step_size: int = 10
    lr_gamma: float = 0.1

    # Augmentation
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.2
    random_crop_scale: tuple = (0.8, 1.0)

    # Class imbalance
    use_class_weights: bool = True

    # Validation
    val_split: float = 0.2
    early_stopping_patience: int = 7


@dataclass
class InferenceConfig:
    """Inference / real-time detection configuration."""
    checkpoint_path: str = "./output/checkpoints/best_model.pth"
    device: str = "cuda"  # "cuda" or "cpu"
    confidence_threshold: float = 0.5
    clip_duration_sec: float = 2.0
    crop_size: int = 224

    # Webcam
    camera_index: int = 0
    display_width: int = 1280
    display_height: int = 720

    # Loitering / Prolonged Stay detection
    loiter_time_threshold_sec: float = 120.0  # 2 minutes
    loiter_zone: Optional[list] = None        # Polygon [(x,y), ...] or None = full frame

    # Person detection (YOLOv8)
    yolo_model: str = "yolov8n.pt"
    person_conf_threshold: float = 0.5


# ──────────────────────────────────────────────────────────────────────
# Dataset configurations  – EDIT THESE PATHS to match your local setup
# ──────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = [
    DatasetConfig(
        name="rwf2000",
        root_dir="./data/RWF-2000",
        label_map={
            "Fight": CLASS_TO_IDX["fighting"],
            "NonFight": CLASS_TO_IDX["normal"],
        },
        video_ext=".avi",
    ),
    
    DatasetConfig(
        name="ucf_crime",
        root_dir="../data/UCF-Crime",
        label_map={
            "Fighting": CLASS_TO_IDX["fighting"],
            "Assault": CLASS_TO_IDX["fighting"],
            "Vandalism": CLASS_TO_IDX["vandalism"],
            "Normal": CLASS_TO_IDX["normal"],
            # Add more UCF-Crime categories as needed
        },
        video_ext=".mp4",
    ),
    DatasetConfig(
        name="xd_violence",
        root_dir="./data/XD-Violence",
        label_map={
            "Fighting": CLASS_TO_IDX["fighting"],
            "Riot": CLASS_TO_IDX["fighting"],
            "Abuse": CLASS_TO_IDX["harassment"],
            "Normal": CLASS_TO_IDX["normal"],
        },
        video_ext=".mp4",
    ),
    DatasetConfig(
        name="ur_fall",
        root_dir="./data/UR-Fall",
        label_map={
            "Fall": CLASS_TO_IDX["faint"],
            "NonFall": CLASS_TO_IDX["normal"],
        },
        video_ext=".avi",
    ),
]
