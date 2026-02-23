"""
Dataset module for loading and preprocessing video clips from multiple
surveillance / action-recognition datasets into a unified format for SlowFast.

Supported dataset structures
─────────────────────────────
1. Folder-based (RWF-2000, UR-Fall):
       root_dir/
         train/
           ClassName1/  video1.avi  video2.avi ...
           ClassName2/  ...
         val/
           ClassName1/  ...

2. Annotation-file-based (UCF-Crime, XD-Violence):
       root_dir/
         videos/       video1.mp4  video2.mp4 ...
         annotations/  train.txt   test.txt
       Each line in the annotation file:  <video_path> <label>
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import av
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms as T

from configs.config import (
    DatasetConfig, TrainConfig, ACTION_CLASSES, CLASS_TO_IDX, NUM_CLASSES
)


# ══════════════════════════════════════════════════════════════════════
#  Video I/O utilities
# ══════════════════════════════════════════════════════════════════════

def decode_video_pyav(video_path: str, num_frames: int, fps: Optional[float] = None) -> np.ndarray:
    """
    Decode *num_frames* uniformly-sampled RGB frames from a video file using PyAV.
    Returns shape (T, H, W, 3) uint8.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        # Estimate from duration
        duration = float(stream.duration * stream.time_base)
        avg_fps = float(stream.average_rate) if stream.average_rate else 25.0
        total_frames = int(duration * avg_fps)

    total_frames = max(total_frames, num_frames)

    # Compute indices to sample uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    indices_set = set(indices.tolist())

    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices_set:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        frame_idx += 1
        if len(frames) == num_frames:
            break

    container.close()

    # If we got fewer frames than needed (short video), duplicate last frame
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames, axis=0)


# ══════════════════════════════════════════════════════════════════════
#  SlowFast-specific frame packing
# ══════════════════════════════════════════════════════════════════════

def pack_frames_slowfast(
    frames: np.ndarray,
    num_frames_slow: int = 8,
    num_frames_fast: int = 32,
    crop_size: int = 224,
    spatial_transform: Optional[Callable] = None,
) -> List[torch.Tensor]:
    """
    Given T frames (T, H, W, 3), produce the two-pathway input for SlowFast:
      - slow_pathway: (1, 3, num_frames_slow, crop_size, crop_size)
      - fast_pathway: (1, 3, num_frames_fast, crop_size, crop_size)

    Returns [slow_tensor, fast_tensor] each of shape (3, T, H, W).
    """
    T_total = frames.shape[0]

    # --- Sample slow & fast indices ---
    slow_indices = np.linspace(0, T_total - 1, num_frames_slow, dtype=np.int64)
    fast_indices = np.linspace(0, T_total - 1, num_frames_fast, dtype=np.int64)

    slow_frames = frames[slow_indices]  # (Ts, H, W, 3)
    fast_frames = frames[fast_indices]  # (Tf, H, W, 3)

    # --- Apply spatial transforms (resize + crop + normalize) ---
    slow_tensor = _frames_to_tensor(slow_frames, crop_size, spatial_transform)
    fast_tensor = _frames_to_tensor(fast_frames, crop_size, spatial_transform)

    return [slow_tensor, fast_tensor]


def _frames_to_tensor(
    frames: np.ndarray,
    crop_size: int,
    spatial_transform: Optional[Callable] = None,
) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 → (3, T, crop_size, crop_size) float tensor."""
    num_t, H, W, C = frames.shape

    # Resize shortest side to crop_size, then center-crop
    tensors = []
    for t in range(num_t):
        img = torch.from_numpy(frames[t]).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        # Resize
        img = T.functional.resize(img, [crop_size, crop_size], antialias=True)
        tensors.append(img)

    tensor = torch.stack(tensors, dim=1)  # (3, num_t, H, W)

    if spatial_transform is not None:
        tensor = spatial_transform(tensor)

    # Normalize with Kinetics mean/std
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


# ══════════════════════════════════════════════════════════════════════
#  Spatial augmentation for training
# ══════════════════════════════════════════════════════════════════════

class SpatialAugmentation:
    """
    Augmentations applied consistently across all frames of a clip.
    Operates on tensor of shape (3, T, H, W).
    """

    def __init__(self, cfg: TrainConfig, is_train: bool = True):
        self.is_train = is_train
        self.cfg = cfg

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (3, T, H, W)
        if not self.is_train:
            return tensor

        # Random horizontal flip (applied to all frames consistently)
        if random.random() < self.cfg.horizontal_flip_prob:
            tensor = torch.flip(tensor, dims=[-1])

        # Random color jitter (applied frame-by-frame but with same params)
        if self.cfg.color_jitter > 0:
            jitter = self.cfg.color_jitter
            brightness = 1.0 + random.uniform(-jitter, jitter)
            contrast = 1.0 + random.uniform(-jitter, jitter)
            tensor = torch.clamp(tensor * brightness, 0, 1)
            mean = tensor.mean(dim=(0, 2, 3), keepdim=True)
            tensor = torch.clamp((tensor - mean) * contrast + mean, 0, 1)

        return tensor


# ══════════════════════════════════════════════════════════════════════
#  Unified Video Dataset
# ══════════════════════════════════════════════════════════════════════

class ActionVideoDataset(Dataset):
    """
    Unified dataset that loads video clips and returns SlowFast input pairs.

    Supports two modes of discovering videos:
    1. Folder-based: root_dir/{split}/{class_name}/video.ext
    2. Annotation-based: reads an annotation .txt file with lines
       "relative/path/to/video.ext <label_string>"
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        train_cfg: TrainConfig,
        split: str = "train",
        is_train: bool = True,
        annotation_file: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.is_train = is_train
        self.spatial_aug = SpatialAugmentation(train_cfg, is_train)

        # Total frames to decode (enough for both pathways)
        self.num_frames_decode = max(train_cfg.num_frames_fast, train_cfg.num_frames_slow * 4)

        # Build list of (video_path, label_idx)
        self.samples: List[Tuple[str, int]] = []

        if annotation_file and os.path.isfile(annotation_file):
            self._load_from_annotation(annotation_file)
        else:
            self._load_from_folders(split)

        print(f"[{dataset_cfg.name}/{split}] Loaded {len(self.samples)} clips "
              f"({self._class_distribution()})")

    # ── discovery helpers ──

    def _load_from_folders(self, split: str):
        """Folder-based: root_dir/{split}/{class_name}/video.ext"""
        split_dir = os.path.join(self.dataset_cfg.root_dir, split)
        if not os.path.isdir(split_dir):
            # Try without split subfolder (flat structure)
            split_dir = self.dataset_cfg.root_dir

        for class_name, label_idx in self.dataset_cfg.label_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.dataset_cfg.video_ext):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, label_idx))

    def _load_from_annotation(self, annotation_file: str):
        """Annotation-file-based: each line → 'rel_path label_string'"""
        root = self.dataset_cfg.root_dir
        with open(annotation_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                rel_path, label_str = parts
                if label_str in self.dataset_cfg.label_map:
                    label_idx = self.dataset_cfg.label_map[label_str]
                    video_path = os.path.join(root, rel_path)
                    if os.path.isfile(video_path):
                        self.samples.append((video_path, label_idx))

    def _class_distribution(self) -> str:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        parts = [f"{ACTION_CLASSES[k]}:{v}" for k, v in sorted(counts.items())]
        return ", ".join(parts)

    # ── __getitem__ ──

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        try:
            frames = decode_video_pyav(video_path, self.num_frames_decode)
        except Exception as e:
            print(f"⚠ Error decoding {video_path}: {e}. Returning zeros.")
            h = w = self.train_cfg.crop_size
            frames = np.zeros((self.num_frames_decode, h, w, 3), dtype=np.uint8)

        slow_tensor, fast_tensor = pack_frames_slowfast(
            frames,
            num_frames_slow=self.train_cfg.num_frames_slow,
            num_frames_fast=self.train_cfg.num_frames_fast,
            crop_size=self.train_cfg.crop_size,
            spatial_transform=self.spatial_aug,
        )

        label_tensor = torch.tensor(label, dtype=torch.long)
        return [slow_tensor, fast_tensor], label_tensor


# ══════════════════════════════════════════════════════════════════════
#  Build combined DataLoaders
# ══════════════════════════════════════════════════════════════════════

def build_datasets(
    dataset_configs: List[DatasetConfig],
    train_cfg: TrainConfig,
) -> Tuple[Dataset, Dataset]:
    """
    Build combined train and val datasets from all configured sources.
    If datasets don't have explicit val splits, we create a random split.
    """
    train_datasets = []
    val_datasets = []

    for dcfg in dataset_configs:
        if not os.path.isdir(dcfg.root_dir):
            print(f"⚠ Dataset directory not found: {dcfg.root_dir} — skipping {dcfg.name}")
            continue

        # Check if train/val split folders exist
        has_split_folders = (
            os.path.isdir(os.path.join(dcfg.root_dir, "train")) and
            os.path.isdir(os.path.join(dcfg.root_dir, "val"))
        )

        if has_split_folders:
            train_ds = ActionVideoDataset(dcfg, train_cfg, split="train", is_train=True)
            val_ds = ActionVideoDataset(dcfg, train_cfg, split="val", is_train=False)
        else:
            # Create a single dataset and we'll split later
            full_ds = ActionVideoDataset(dcfg, train_cfg, split="train", is_train=True)
            n_val = int(len(full_ds) * train_cfg.val_split)
            n_train = len(full_ds) - n_val
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    if not train_datasets:
        raise RuntimeError("No datasets were loaded! Check your data paths in configs/config.py")

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)

    print(f"\n{'='*60}")
    print(f"Combined: {len(combined_train)} train clips, {len(combined_val)} val clips")
    print(f"{'='*60}\n")

    return combined_train, combined_val


def build_dataloaders(
    dataset_configs: List[DatasetConfig],
    train_cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders with optional class-weighted sampling."""
    train_dataset, val_dataset = build_datasets(dataset_configs, train_cfg)

    # --- Weighted sampler for class imbalance ---
    sampler = None
    shuffle = True
    if train_cfg.use_class_weights:
        labels = []
        for i in range(len(train_dataset)):
            # ConcatDataset: access underlying dataset
            ds = train_dataset
            if hasattr(ds, "datasets"):
                # Walk through ConcatDataset to get the label
                cumulative = 0
                for sub_ds in ds.datasets:
                    if i < cumulative + len(sub_ds):
                        local_idx = i - cumulative
                        if hasattr(sub_ds, "samples"):
                            _, lbl = sub_ds.samples[local_idx]
                        elif hasattr(sub_ds, "dataset"):
                            # random_split subset
                            real_idx = sub_ds.indices[local_idx]
                            _, lbl = sub_ds.dataset.samples[real_idx]
                        else:
                            lbl = 0
                        labels.append(lbl)
                        break
                    cumulative += len(sub_ds)
            else:
                _, lbl = ds[i]
                labels.append(lbl.item() if isinstance(lbl, torch.Tensor) else lbl)

        class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        class_counts = np.maximum(class_counts, 1.0)  # avoid division by zero
        weights_per_class = 1.0 / class_counts
        sample_weights = [weights_per_class[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    def slowfast_collate(batch):
        """Custom collate: batch list of ([slow, fast], label)."""
        slow_clips = torch.stack([item[0][0] for item in batch])
        fast_clips = torch.stack([item[0][1] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return [slow_clips, fast_clips], labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=slowfast_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=slowfast_collate,
        pin_memory=True,
    )

    return train_loader, val_loader
