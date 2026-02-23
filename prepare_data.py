"""
prepare_data.py — Download helper and dataset organizer.

This script helps you organize your downloaded datasets into the expected
folder structure for training. Since most of these datasets require manual
download (registration / license agreement), this script focuses on
organizing and verifying the data after you've downloaded it.

Expected final structure after running this script:
    data/
    ├── RWF-2000/
    │   ├── train/
    │   │   ├── Fight/       *.avi
    │   │   └── NonFight/    *.avi
    │   └── val/
    │       ├── Fight/       *.avi
    │       └── NonFight/    *.avi
    ├── UCF-Crime/
    │   ├── videos/          *.mp4
    │   └── annotations/
    │       ├── train.txt
    │       └── test.txt
    ├── XD-Violence/
    │   ├── videos/          *.mp4
    │   └── annotations/
    │       ├── train.txt
    │       └── test.txt
    └── UR-Fall/
        ├── train/
        │   ├── Fall/        *.avi
        │   └── NonFall/     *.avi
        └── val/
            ├── Fall/        *.avi
            └── NonFall/     *.avi

Usage:
    # Verify your data setup
    python prepare_data.py --verify

    # Organize RWF-2000 from a downloaded zip
    python prepare_data.py --organize rwf2000 --source /path/to/downloaded/RWF-2000

    # Create train/val split for UR-Fall
    python prepare_data.py --organize ur_fall --source /path/to/UR-Fall --val_ratio 0.2

    # Generate annotation files for UCF-Crime
    python prepare_data.py --organize ucf_crime --source /path/to/UCF-Crime
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import DATASET_CONFIGS, ACTION_CLASSES, CLASS_TO_IDX


DATA_DIR = "./data"


def verify_datasets():
    """Check which datasets are properly set up."""
    print(f"\n{'='*60}")
    print(f"  Dataset Verification")
    print(f"{'='*60}\n")

    total_videos = 0
    for dcfg in DATASET_CONFIGS:
        print(f"📂 {dcfg.name}: {dcfg.root_dir}")
        if not os.path.isdir(dcfg.root_dir):
            print(f"   ✗ Directory not found!\n")
            continue

        # Count videos
        count = 0
        for root, dirs, files in os.walk(dcfg.root_dir):
            for f in files:
                if f.lower().endswith((dcfg.video_ext, ".mp4", ".avi", ".mkv")):
                    count += 1

        total_videos += count
        if count > 0:
            print(f"   ✓ Found {count} video files")
        else:
            print(f"   ⚠ No video files found (expected *{dcfg.video_ext})")

        # Check for train/val split
        has_train = os.path.isdir(os.path.join(dcfg.root_dir, "train"))
        has_val = os.path.isdir(os.path.join(dcfg.root_dir, "val"))
        has_annotations = os.path.isdir(os.path.join(dcfg.root_dir, "annotations"))

        if has_train and has_val:
            train_count = sum(1 for _ in Path(dcfg.root_dir, "train").rglob(f"*{dcfg.video_ext}"))
            val_count = sum(1 for _ in Path(dcfg.root_dir, "val").rglob(f"*{dcfg.video_ext}"))
            print(f"   ✓ Train/Val split: {train_count} train, {val_count} val")
        elif has_annotations:
            print(f"   ✓ Annotation files found")
        else:
            print(f"   ⚠ No train/val split or annotation files found")

        # Check class folders
        for class_name in dcfg.label_map.keys():
            for split in ["train", "val", ""]:
                class_dir = os.path.join(dcfg.root_dir, split, class_name) if split else \
                            os.path.join(dcfg.root_dir, class_name)
                if os.path.isdir(class_dir):
                    n = sum(1 for f in os.listdir(class_dir)
                            if f.lower().endswith((dcfg.video_ext, ".mp4", ".avi")))
                    if n > 0:
                        print(f"   ✓ {split + '/' if split else ''}{class_name}: {n} videos")

        print()

    print(f"{'─'*60}")
    print(f"Total videos found: {total_videos}")
    if total_videos == 0:
        print(f"\n⚠ No datasets found! Please download and organize datasets first.")
        print(f"  See the README for download links and instructions.")
    print()


def organize_folder_dataset(
    source_dir: str,
    target_dir: str,
    class_folders: Dict[str, str],
    val_ratio: float = 0.2,
    video_ext: str = ".avi",
):
    """
    Organize a folder-based dataset into train/val splits.

    Args:
        source_dir: Path to the downloaded dataset
        target_dir: Where to create the organized structure
        class_folders: Mapping of source folder names → target class names
        val_ratio: Fraction of data to use for validation
        video_ext: Video file extension to look for
    """
    os.makedirs(target_dir, exist_ok=True)

    for src_class, tgt_class in class_folders.items():
        # Find source videos
        src_path = None
        for candidate in [
            os.path.join(source_dir, src_class),
            os.path.join(source_dir, "train", src_class),
            os.path.join(source_dir, "videos", src_class),
        ]:
            if os.path.isdir(candidate):
                src_path = candidate
                break

        if src_path is None:
            print(f"  ⚠ Source folder not found for class '{src_class}'")
            continue

        videos = [f for f in os.listdir(src_path)
                   if f.lower().endswith((video_ext, ".mp4", ".avi", ".mkv"))]
        random.shuffle(videos)

        n_val = int(len(videos) * val_ratio)
        val_videos = videos[:n_val]
        train_videos = videos[n_val:]

        # Create directories
        train_dir = os.path.join(target_dir, "train", tgt_class)
        val_dir = os.path.join(target_dir, "val", tgt_class)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy/symlink videos
        for v in train_videos:
            src = os.path.join(src_path, v)
            dst = os.path.join(train_dir, v)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        for v in val_videos:
            src = os.path.join(src_path, v)
            dst = os.path.join(val_dir, v)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        print(f"  ✓ {tgt_class}: {len(train_videos)} train, {len(val_videos)} val")


def generate_annotation_file(
    video_dir: str,
    output_file: str,
    label_map: Dict[str, str],
    video_ext: str = ".mp4",
):
    """
    Generate an annotation file for datasets organized as:
        video_dir/<ClassName>/video.mp4
    Output: each line is "relative/path.mp4 ClassName"
    """
    lines = []
    for folder_name, label_name in label_map.items():
        folder_path = os.path.join(video_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((video_ext, ".mp4", ".avi")):
                rel_path = os.path.join(folder_name, fname)
                lines.append(f"{rel_path} {label_name}")

    random.shuffle(lines)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Wrote {len(lines)} entries to {output_file}")


def organize_rwf2000(source_dir: str):
    """Organize RWF-2000 dataset."""
    print("\n📂 Organizing RWF-2000...")
    target = os.path.join(DATA_DIR, "RWF-2000")
    organize_folder_dataset(
        source_dir, target,
        class_folders={"Fight": "Fight", "NonFight": "NonFight"},
        val_ratio=0.2, video_ext=".avi",
    )


def organize_ucf_crime(source_dir: str):
    """Organize UCF-Crime dataset."""
    print("\n📂 Organizing UCF-Crime...")
    target = os.path.join(DATA_DIR, "UCF-Crime")
    videos_dir = os.path.join(target, "videos")
    annot_dir = os.path.join(target, "annotations")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    label_map = {
        "Fighting": "Fighting",
        "Assault": "Assault",
        "Vandalism": "Vandalism",
        "Normal_Videos_event": "Normal",
    }

    # Copy/link videos
    for src_class in label_map.keys():
        src_path = os.path.join(source_dir, src_class)
        if not os.path.isdir(src_path):
            src_path = os.path.join(source_dir, "Videos", src_class)
        if not os.path.isdir(src_path):
            continue
        tgt_path = os.path.join(videos_dir, src_class)
        os.makedirs(tgt_path, exist_ok=True)
        for f in os.listdir(src_path):
            if f.lower().endswith((".mp4", ".avi")):
                src = os.path.join(src_path, f)
                dst = os.path.join(tgt_path, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)

    # Generate annotation files
    all_lines = []
    for folder_name, label_name in label_map.items():
        folder_path = os.path.join(videos_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".mp4", ".avi")):
                rel_path = os.path.join("videos", folder_name, fname)
                all_lines.append(f"{rel_path} {label_name}")

    random.shuffle(all_lines)
    split_idx = int(len(all_lines) * 0.8)

    with open(os.path.join(annot_dir, "train.txt"), "w") as f:
        f.write("\n".join(all_lines[:split_idx]))
    with open(os.path.join(annot_dir, "test.txt"), "w") as f:
        f.write("\n".join(all_lines[split_idx:]))

    print(f"  ✓ Train: {split_idx}, Test: {len(all_lines) - split_idx}")


def organize_ur_fall(source_dir: str):
    """Organize UR-Fall dataset."""
    print("\n📂 Organizing UR-Fall...")
    target = os.path.join(DATA_DIR, "UR-Fall")
    organize_folder_dataset(
        source_dir, target,
        class_folders={"Fall": "Fall", "NonFall": "NonFall", "ADL": "NonFall"},
        val_ratio=0.2, video_ext=".avi",
    )


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation")
    parser.add_argument("--verify", action="store_true", help="Verify dataset setup")
    parser.add_argument("--organize", type=str, choices=["rwf2000", "ucf_crime", "ur_fall", "xd_violence"],
                        help="Organize a specific dataset")
    parser.add_argument("--source", type=str, help="Source directory of downloaded dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    if args.verify or (not args.organize):
        verify_datasets()
        return

    if not args.source:
        print("✗ --source required when using --organize")
        return

    if args.organize == "rwf2000":
        organize_rwf2000(args.source)
    elif args.organize == "ucf_crime":
        organize_ucf_crime(args.source)
    elif args.organize == "ur_fall":
        organize_ur_fall(args.source)
    else:
        print(f"⚠ Organizer for {args.organize} not yet implemented. Contribute it!")

    print("\n✓ Done! Run `python prepare_data.py --verify` to check.")


if __name__ == "__main__":
    main()
