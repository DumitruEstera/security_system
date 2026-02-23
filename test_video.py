"""
test_video.py — Classify a single video using a trained SlowFast model.

Usage:
    python test_video.py --video path/to/video.avi --checkpoint output/checkpoints/best_model.pth
    python test_video.py --video fight_clip.avi                    # Uses default checkpoint path
    python test_video.py --video video.mp4 --device cpu            # Force CPU
    python test_video.py --video video.mp4 --num_classes 3         # If trained with 3 classes
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ACTION_CLASSES, NUM_CLASSES, IDX_TO_CLASS
from data.dataset import decode_video_pyav, pack_frames_slowfast
from models.slowfast_model import load_model_for_inference


def classify_video(
    video_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    num_classes: int = None,
) -> dict:
    """
    Classify a single video file.

    Returns dict with predicted class, confidence, and all class probabilities.
    """
    if num_classes is None:
        num_classes = NUM_CLASSES

    # Load model
    model = load_model_for_inference(checkpoint_path, num_classes=num_classes, device=device)

    # Decode video frames
    num_frames_decode = 64
    print(f"📹 Decoding video: {video_path}")
    frames = decode_video_pyav(video_path, num_frames_decode)
    print(f"   Decoded {frames.shape[0]} frames, resolution {frames.shape[1]}x{frames.shape[2]}")

    # Pack into SlowFast format
    slow_tensor, fast_tensor = pack_frames_slowfast(
        frames,
        num_frames_slow=8,
        num_frames_fast=32,
        crop_size=224,
        spatial_transform=None,
    )

    # Add batch dimension and move to device
    slow_tensor = slow_tensor.unsqueeze(0).to(device)
    fast_tensor = fast_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model([slow_tensor, fast_tensor])
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Get prediction
    confidence, pred_idx = probs.max(dim=0)
    pred_class = IDX_TO_CLASS.get(pred_idx.item(), f"class_{pred_idx.item()}")

    # Build results
    all_probs = {
        IDX_TO_CLASS.get(i, f"class_{i}"): float(probs[i])
        for i in range(num_classes)
    }

    return {
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "all_probabilities": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a video with trained SlowFast model")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--checkpoint", type=str, default="./output/checkpoints/best_model.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected if not set)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes the model was trained with (auto-detected from data/train/)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"✗ Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Classify
    result = classify_video(args.video, args.checkpoint, device, args.num_classes)

    # Display results
    print(f"\n{'='*50}")
    print(f"  Video:      {args.video}")
    print(f"  Prediction: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"{'='*50}")
    print(f"  All class probabilities:")
    sorted_probs = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs:
        bar = "█" * int(prob * 30)
        marker = " ◄" if class_name == result["predicted_class"] else ""
        print(f"    {class_name:<22} {prob:>6.1%}  {bar}{marker}")
    print()


if __name__ == "__main__":
    main()