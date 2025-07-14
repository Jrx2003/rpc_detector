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

## Stage B: Knowledge Distillation with Ensemble Teacher

### Overview

Knowledge distillation is a technique where a smaller "student" model learns from a larger, more powerful "teacher" model. In our implementation, we use an ensemble of two large models (YOLOv8x + YOLOv8l) as teachers to distill knowledge into our compact student model (YOLOv8m from Stage A).

### Architecture

```
    Teacher Models (Ensemble)
    ┌─────────────────────────────┐
    │     YOLOv8x (Teacher 1)     │
    │         ↓                   │
    │    Teacher Outputs 1        │
    └─────────────────────────────┘
                 ↓
    ┌─────────────────────────────┐
    │      Average Outputs        │ ← Ensemble
    │    (Logits + BBox Reg)      │
    └─────────────────────────────┘
                 ↓
    ┌─────────────────────────────┐
    │     YOLOv8l (Teacher 2)     │
    │         ↓                   │
    │    Teacher Outputs 2        │
    └─────────────────────────────┘
                 ↓
           Knowledge Transfer
                 ↓
    ┌─────────────────────────────┐
    │    YOLOv8m (Student)        │
    │    From Stage A + Focal     │
    └─────────────────────────────┘
```

### Knowledge Distillation Process

1. **Teacher Forward Pass**: The ensemble teacher (YOLOv8x + YOLOv8l) processes input images and produces averaged outputs (no gradients computed)

2. **Student Forward Pass**: The student model (YOLOv8m) processes the same images and produces its own outputs

3. **Loss Calculation**:
   - `loss_det`: Original YOLO detection loss (student predictions vs. ground truth)
   - `loss_kd`: Knowledge distillation loss (student predictions vs. teacher predictions)
   - `total_loss = α * loss_det + (1-α) * loss_kd`

4. **Knowledge Distillation Loss Components**:
   - **KL Divergence Loss**: `KL(log_softmax(student/T), softmax(teacher/T)) * T²`
   - **MSE Loss**: `MSE(student_bbox, teacher_bbox)`

### Key Parameters

- **Temperature (T)**: Controls the softness of probability distributions
  - Higher T → softer distributions → easier knowledge transfer
  - Lower T → harder distributions → more focused on confident predictions
  - Default: T = 4.0

- **Alpha (α)**: Balances detection loss and distillation loss
  - α = 0.5 → Equal weighting of both losses
  - Higher α → More emphasis on ground truth
  - Lower α → More emphasis on teacher knowledge

### Expected Benefits

1. **Improved Accuracy**: Student learns from multiple teacher perspectives
2. **Better Generalization**: Ensemble knowledge provides robustness
3. **Maintained Efficiency**: Student model remains compact (YOLOv8m)
4. **Enhanced Tail-Class Performance**: Combined with Stage A focal loss benefits

### Usage

```bash
# Train with knowledge distillation
python train_distill.py --teacher_runs yolov8x.pt yolov8l.pt --student_ckpt runs/focal/weights/best.pt --epochs 60 --batch 8 --evaluate
```

## Files Description

- `requirements.txt`: Python dependencies
- `scripts/rpc_to_yolo.py`: Dataset conversion script
- `train.py`: YOLOv8 baseline training script
- `train_focal.py`: YOLOv8 focal loss training script (Stage A)
- `train_distill.py`: Knowledge distillation training script (Stage B)
- `distill/ensemble_teacher.py`: Ensemble teacher implementation
- `hyp_focal.yaml`: Focal loss hyperparameters configuration
- `tools/gen_class_weights.py`: Dynamic class weights generation
- `data/RPC/rpc.yaml`: YOLO dataset configuration file (auto-generated)
- `data/RPC/class_weights.npy`: Dynamic class weights (auto-generated) 