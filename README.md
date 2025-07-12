# RPC Detector - YOLOv8 Baseline

This repository contains code for training a YOLOv8 baseline model on the RPC (Retail Product Checkout) dataset.

## Dataset

The RPC dataset contains 200 product categories with 24,000 images and 294,333 annotations. The dataset is already included in the `rpc/` directory.

Dataset source: [RPC Dataset](https://github.com/retail-product-checkout/RPC-Dataset)

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n py38 python=3.8
conda activate py38
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Data Directory Structure

The expected directory structure:
```
rpc_detector/
├── rpc/
│   ├── instances_test2019.json
│   └── test2019/
│       └── *.jpg
├── data/
│   └── RPC/
│       ├── raw/           # Will be created from rpc/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   └── rpc_to_yolo.py
└── train.py
```

## Usage

### 1. Convert RPC Dataset to YOLO Format

```bash
python scripts/rpc_to_yolo.py --root rpc/
```

This will:
- Convert COCO format annotations to YOLO format
- Split data into train/val/test (70/20/10)
- Create `data/RPC/rpc.yaml` with class names and paths

### 2. Train YOLOv8 Model

```bash
python train.py
```

This will:
- Download YOLOv8m pre-trained weights
- Fine-tune on RPC dataset for 50 epochs
- Save results to `runs/baseline/`
- Log training progress to TensorBoard

### 3. View Training Results

```bash
tensorboard --logdir runs/baseline
```

## Model Configuration

- **Model**: YOLOv8m (medium)
- **Input Size**: 640x640
- **Epochs**: 50
- **Batch Size**: 16
- **Device**: GPU (device=0)

## Results

After training, the model will be evaluated on the test split and save:
- Model checkpoints in `runs/baseline/weights/`
- Training logs and metrics
- Sample prediction images in `runs/baseline/samples/`

## Files Description

- `requirements.txt`: Python dependencies
- `scripts/rpc_to_yolo.py`: Dataset conversion script
- `train.py`: YOLOv8 training script
- `data/RPC/rpc.yaml`: YOLO dataset configuration file (auto-generated) 