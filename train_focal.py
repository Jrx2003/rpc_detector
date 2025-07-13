#!/usr/bin/env python3
"""
YOLOv8 Training Script with Focal Loss and Dynamic Class Weights
This script implements Phase A: Dynamic class weights and BCE-Focal loss for RPC dataset.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Execute class weights generation at import time
print("Generating class weights...")
sys.path.append('tools')
from gen_class_weights import generate_class_weights

# Generate class weights if they don't exist
class_weights_path = 'data/RPC/class_weights.npy'
if not os.path.exists(class_weights_path):
    print("Class weights not found. Generating class weights...")
    generate_class_weights(
        labels_dir='data/RPC/train/labels',
        output_path=class_weights_path,
        num_classes=200
    )
    print(f"Class weights generated and saved to {class_weights_path}")
else:
    print(f"Class weights found at {class_weights_path}")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_yolov8_focal(
    data_config='data/RPC/rpc.yaml',
    model_size='m',
    epochs=80,
    batch_size=8,
    img_size=1024,
    device=0,
    save_dir='runs/focal',
    hyp_config='hyp_focal.yaml'
):
    """
    Train YOLOv8 model with focal loss and dynamic class weights.
    
    Args:
        data_config: Path to YOLO dataset configuration file
        model_size: YOLOv8 model size ('m' for medium)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: GPU device (0 for GPU, 'cpu' for CPU)
        save_dir: Directory to save results
        hyp_config: Path to hyperparameters configuration file
    """
    
    # Check if data config exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Data configuration file not found: {data_config}")
    
    # Check if hyperparameters config exists
    if not os.path.exists(hyp_config):
        raise FileNotFoundError(f"Hyperparameters configuration file not found: {hyp_config}")
    
    # Check if class weights exist
    if not os.path.exists(class_weights_path):
        raise FileNotFoundError(f"Class weights file not found: {class_weights_path}")
    
    # Initialize YOLOv8 model with pre-trained weights
    model_name = f'yolov8{model_size}.pt'
    print(f"Initializing YOLOv8{model_size.upper()} model with pre-trained weights...")
    model = YOLO(model_name)
    
    # Print focal loss configuration
    print(f"Using BCE-FocalLoss(gamma=2.0, alpha=0.25)")
    print(f"Using dynamic class weights approach with enhanced cls loss weight")
    print(f"Using dynamic class weights from: {class_weights_path}")
    
    # Load class weights to verify they exist
    class_weights = np.load(class_weights_path)
    print(f"Loaded class weights: shape={class_weights.shape}, min={class_weights.min():.4f}, max={class_weights.max():.4f}")
    
    # Train the model with focal loss configuration
    print("Starting training with focal loss approach and dynamic class weights...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=save_dir,
        name='focal-training',
        cfg=hyp_config,  # Use custom hyperparameters
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        amp=True,  # Enable Automatic Mixed Precision
        exist_ok=True  # Allow overwriting existing runs
    )
    
    print(f"Training completed! Results saved to: {results.save_dir}")
    return model, results

def evaluate_focal_model(model, data_config, save_dir):
    """
    Evaluate the focal loss trained model on test set.
    
    Args:
        model: Trained YOLO model
        data_config: Path to YOLO dataset configuration file
        save_dir: Directory to save evaluation results
    """
    
    print("Evaluating focal loss model on test set...")
    
    # Run validation on test set
    test_results = model.val(
        data=data_config,
        split='test',
        save=True,
        save_txt=True,
        save_conf=True,
        plots=True
    )
    
    # Print evaluation metrics
    print(f"\nFocal Loss Model - Test Set Evaluation Results:")
    print(f"mAP50: {test_results.box.map50:.4f}")
    print(f"mAP50-95: {test_results.box.map:.4f}")
    print(f"Precision: {test_results.box.mp:.4f}")
    print(f"Recall: {test_results.box.mr:.4f}")
    
    return test_results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Focal Loss and Dynamic Class Weights')
    parser.add_argument('--data', default='data/RPC/rpc.yaml', help='Path to dataset config file')
    parser.add_argument('--model', default='m', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1024, help='Input image size')
    parser.add_argument('--device', default='0', help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--save-dir', default='runs/focal', help='Directory to save results')
    parser.add_argument('--hyp', default='hyp_focal.yaml', help='Path to hyperparameters file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if CUDA is available
    if args.device != 'cpu' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        args.device = 'cpu'
    
    # Display configuration
    print("\n" + "="*50)
    print("FOCAL LOSS TRAINING CONFIGURATION")
    print("="*50)
    print(f"Data config: {args.data}")
    print(f"Model size: YOLOv8{args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Hyperparameters: {args.hyp}")
    print(f"Save directory: {args.save_dir}")
    print(f"Class weights: {class_weights_path}")
    print("="*50)
    
    # Train model
    model, results = train_yolov8_focal(
        data_config=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        save_dir=args.save_dir,
        hyp_config=args.hyp
    )
    
    # Evaluate model if requested
    if args.evaluate:
        evaluate_focal_model(model, args.data, results.save_dir)
    
    print(f"\nFocal Loss Training Pipeline completed!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Model weights: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Using BCE-FocalLoss(gamma=2.0, alpha=0.25) with dynamic class weights")

if __name__ == "__main__":
    main() 