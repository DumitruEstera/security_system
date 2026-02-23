# Security Surveillance Action Recognition System

A real-time security monitoring system using **SlowFast** (video action recognition), **YOLOv8** (person detection), and **DeepSORT** (person tracking) to detect threats in surveillance footage.

## Detected Scenarios

| Scenario | Method | Dataset Sources |
|---|---|---|
| Vandalism | SlowFast action recognition | UCF-Crime |
| Harassment | SlowFast action recognition | XD-Violence |
| Physical altercation / Fighting | SlowFast action recognition | RWF-2000, UCF-Crime, XD-Violence |
| Faint / Collapse | SlowFast action recognition | UR Fall Detection |
| Handling dangerous objects | SlowFast action recognition | Custom / UCF-Crime |
| Suspicious gathering | SlowFast action recognition | Custom annotation |
| Prolonged stay / Loitering | YOLO + DeepSORT tracking + time-in-zone rule | No training needed |

## Architecture

```
Webcam / Video Feed
        │
        ├──► YOLOv8 Person Detection
        │         │
        │         ▼
        │    DeepSORT Tracking ──► Loitering Detector (time-in-zone > threshold → Alert)
        │
        ├──► Frame Buffer (64 frames)
        │         │
        │         ▼
        │    SlowFast-R50 (pretrained Kinetics-400, fine-tuned)
        │         │
        │         ▼
        │    Action Classification → Alert if anomaly detected
        │
        ▼
   Composite HUD Display (bounding boxes, alerts, action labels)
```

## Project Structure

```
security_system/
├── configs/
│   └── config.py              # All hyperparameters, paths, class definitions
├── data/
│   └── dataset.py             # Unified video dataset loader for SlowFast
├── models/
│   └── slowfast_model.py      # SlowFast with custom head + freeze/unfreeze
├── utils/
│   ├── metrics.py             # Accuracy, F1, confusion matrix, early stopping
│   └── loitering_detector.py  # Rule-based prolonged stay detection
├── train.py                   # Fine-tuning script (2-phase: frozen → full)
├── inference.py               # Real-time webcam/video inference pipeline
├── prepare_data.py            # Dataset download helper & organizer
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch (visit https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

### 2. Download & Organize Datasets

Download these datasets (most require registration):

| Dataset | Download | What it provides |
|---|---|---|
| **RWF-2000** | [GitHub](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Fight vs NonFight (2000 clips) |
| **UCF-Crime** | [UCF CRCV](https://www.crcv.ucf.edu/projects/real-world/) | 13 anomaly types including Fighting, Vandalism, Assault |
| **XD-Violence** | [GitHub](https://roc-ng.github.io/XD-Violence/) | Large-scale violence (4754 videos) |
| **UR Fall Detection** | [Website](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html) | Fall / non-fall events (70 sequences) |

After downloading, organize with the helper script:

```bash
# Organize RWF-2000
python prepare_data.py --organize rwf2000 --source /path/to/downloaded/RWF-2000

# Organize UCF-Crime
python prepare_data.py --organize ucf_crime --source /path/to/downloaded/UCF-Crime

# Organize UR-Fall
python prepare_data.py --organize ur_fall --source /path/to/downloaded/UR-Fall

# Verify everything is in place
python prepare_data.py --verify
```

### 3. Edit Data Paths

Open `configs/config.py` and update the `root_dir` paths in `DATASET_CONFIGS` to point to your data.

## Training

### Basic Training

```bash
python train.py
```

### With Custom Parameters

```bash
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 5e-4 \
    --freeze_epochs 10 \
    --device cuda
```

### Resume from Checkpoint

```bash
python train.py --resume output/checkpoints/last_model.pth
```

### Training Strategy

The training uses a **two-phase approach**:

1. **Phase 1 (Frozen Backbone):** Only the new classification head is trained for `freeze_backbone_epochs` (default: 5). This lets the head learn to map Kinetics-400 features to your security classes without corrupting the pretrained backbone.

2. **Phase 2 (Full Fine-tuning):** The entire model is unfrozen. The backbone gets a 10× lower learning rate than the head to preserve general features while adapting to the surveillance domain.

Additional techniques: class-weighted sampling, label smoothing, gradient clipping, mixed-precision training (AMP), early stopping.

## Inference

### With Trained Model + Webcam

```bash
python inference.py --source 0 --checkpoint output/checkpoints/best_model.pth
```

### With a Video File

```bash
python inference.py --source surveillance_video.mp4
```

### Demo Mode (No Trained Model Needed)

Test the detection + tracking + loitering pipeline using only YOLO + DeepSORT:

```bash
python inference.py --demo --source 0
```

### Adjust Thresholds

```bash
python inference.py \
    --source 0 \
    --threshold 0.6 \
    --loiter_time 60
```

## Key Configuration (configs/config.py)

| Parameter | Default | Description |
|---|---|---|
| `clip_duration_sec` | 2.0 | Duration of each clip fed to SlowFast |
| `num_frames_slow` | 8 | Frames for SlowFast slow pathway |
| `num_frames_fast` | 32 | Frames for SlowFast fast pathway |
| `crop_size` | 224 | Spatial resolution |
| `batch_size` | 8 | Training batch size (reduce if OOM) |
| `freeze_backbone_epochs` | 5 | Epochs with frozen backbone |
| `loiter_time_threshold_sec` | 120.0 | Seconds before loitering alert |

## Tips for Your Bachelor's Project

1. **Start with demo mode** (`--demo`) to verify YOLO + tracking works with your laptop webcam.
2. **Start small**: train with just RWF-2000 first (binary fight/non-fight), then add datasets.
3. **Reduce batch size** to 2–4 if you get CUDA OOM errors.
4. **Use CPU** if no GPU: `python train.py --device cpu` (will be slow).
5. **For "prolonged stay"**: no training needed — the loitering detector is purely rule-based (detection + tracking + timer).
6. **For classes with no dataset** (suspicious_gathering, dangerous_object): you can record your own short clips with your laptop webcam and add them to the appropriate folders.
7. **Monitor training** with the generated plots in `output/training_curves.png` and `output/confusion_matrix.png`.

## Recording Your Own Training Data

For classes where public datasets are scarce (like `suspicious_gathering`), you can record clips with your webcam:

```python
import cv2
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('suspicious_001.avi', fourcc, 25.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
```

Place recorded clips in the appropriate class folder under `data/`.