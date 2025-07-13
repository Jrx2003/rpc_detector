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

## Stage A: Focal Loss and Dynamic Class Weights

### What is Focal Loss?

Focal Loss is an advanced loss function specifically designed to address class imbalance in object detection tasks. It was introduced in the RetinaNet paper by Lin et al. (2017) to tackle the problem of easy negatives dominating the training process.

### Focal Loss Formula

The Focal Loss formula is:

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` is the predicted probability for the true class
- `α_t` is the weighting factor for class `t` (α = 0.25 for foreground, 1-α for background)
- `γ` (gamma) is the focusing parameter that controls the steepness of the loss curve (γ = 2.0)

### Key Components

1. **Dynamic Class Weights**: Computed using the formula `weight = (median(freq) / freq_i)^0.5` to give higher weights to rare classes.

2. **Focal Loss Approach**:
   - Enhanced classification loss weight (`cls: 1.0`) for better class balance
   - Label smoothing (`label_smoothing: 0.1`) to help with class imbalance
   - Simulates focal loss behavior through hyperparameter adjustments

### Motivation

The RPC dataset contains 200 product categories with varying instance frequencies, creating a long-tail distribution. Standard cross-entropy loss can be dominated by:
- **Easy negatives**: Background pixels that are easily classified
- **Frequent classes**: Classes with many training examples

Focal Loss addresses this by:
- **Down-weighting easy examples**: The `(1 - p_t)^γ` term reduces loss for well-classified examples
- **Up-weighting hard examples**: Focuses training on difficult or misclassified examples
- **Balancing rare classes**: Dynamic class weights ensure rare classes receive adequate attention

### Expected Improvements

With Stage A implementation, we expect:

1. **↑ Tail-class AP**: Improved Average Precision for rare product categories
2. **↑ Overall mAP**: Better overall performance across all classes
3. **↑ Balanced Detection**: More consistent detection across frequent and rare classes
4. **↑ Hard Example Learning**: Better performance on challenging cases

### Usage

```bash
# Train with focal loss and dynamic class weights
python train_focal.py --epochs 80 --batch 8 --imgsz 1024 --evaluate
```

The system automatically:
- Generates class weights from training data
- Applies focal loss approach with enhanced cls loss weight
- Uses dynamic class weights for balanced training
- Implements label smoothing for class imbalance mitigation

## Files Description

- `requirements.txt`: Python dependencies
- `scripts/rpc_to_yolo.py`: Dataset conversion script
- `train.py`: YOLOv8 baseline training script
- `train_focal.py`: YOLOv8 focal loss training script (Stage A)
- `hyp_focal.yaml`: Focal loss hyperparameters configuration
- `tools/gen_class_weights.py`: Dynamic class weights generation
- `data/RPC/rpc.yaml`: YOLO dataset configuration file (auto-generated)
- `data/RPC/class_weights.npy`: Dynamic class weights (auto-generated) 