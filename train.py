#!/usr/bin/env python3
"""
YOLOv8 Training Script for RPC Dataset
Fine-tunes YOLOv8m on the RPC Retail Product Checkout dataset.
"""

import os
import torch
import argparse
from pathlib import Path
import shutil
import random
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_yolov8(
    data_config='data/RPC/rpc.yaml',
    model_size='m',
    epochs=50,
    batch_size=16,
    img_size=640,
    device=0,
    save_dir='runs/baseline',
    project_name='rpc-baseline'
):
    """
    Train YOLOv8 model on RPC dataset.
    
    Args:
        data_config: Path to YOLO dataset configuration file
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: GPU device (0 for GPU, 'cpu' for CPU)
        save_dir: Directory to save results
        project_name: Project name for logging
    """
    
    # Check if data config exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Data configuration file not found: {data_config}")
    
    # Initialize YOLOv8 model with pre-trained weights
    model_name = f'yolov8{model_size}.pt'
    print(f"Initializing YOLOv8{model_size.upper()} model with pre-trained weights...")
    model = YOLO(model_name)
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=save_dir,
        name=project_name,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True
    )
    
    print(f"Training completed! Results saved to: {results.save_dir}")
    return model, results

def evaluate_model(model, data_config, save_dir):
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained YOLO model
        data_config: Path to YOLO dataset configuration file
        save_dir: Directory to save evaluation results
    """
    
    print("Evaluating model on test set...")
    
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
    print(f"\nTest Set Evaluation Results:")
    print(f"mAP50: {test_results.box.map50:.4f}")
    print(f"mAP50-95: {test_results.box.map:.4f}")
    print(f"Precision: {test_results.box.mp:.4f}")
    print(f"Recall: {test_results.box.mr:.4f}")
    
    return test_results

def save_sample_predictions(model, data_config, save_dir, num_samples=5):
    """
    Save sample prediction images.
    
    Args:
        model: Trained YOLO model
        data_config: Path to YOLO dataset configuration file
        save_dir: Directory to save sample images
        num_samples: Number of sample images to save
    """
    
    import yaml
    
    # Load dataset configuration
    with open(data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Get test images directory
    test_images_dir = Path(data_cfg['path']) / data_cfg['test']
    
    if not test_images_dir.exists():
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Get random sample images
    image_files = list(test_images_dir.glob('*.jpg'))
    if len(image_files) < num_samples:
        num_samples = len(image_files)
    
    sample_images = random.sample(image_files, num_samples)
    
    # Create samples directory
    samples_dir = Path(save_dir) / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {num_samples} sample prediction images...")
    
    for i, img_path in enumerate(sample_images):
        # Run prediction
        results = model.predict(source=str(img_path), save=False, conf=0.25)
        
        # Save annotated image
        if results:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Get annotated image
                annotated_img = result.plot()
                
                # Save image
                output_path = samples_dir / f'sample_{i+1}_{img_path.name}'
                cv2.imwrite(str(output_path), annotated_img)
                print(f"Saved sample {i+1}: {output_path}")
            else:
                print(f"No detections in sample {i+1}: {img_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on RPC dataset')
    parser.add_argument('--data', default='data/RPC/rpc.yaml', help='Path to dataset config file')
    parser.add_argument('--model', default='m', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', default='0', help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--save-dir', default='runs/baseline', help='Directory to save results')
    parser.add_argument('--project', default='rpc-baseline', help='Project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample images to save')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if CUDA is available
    if args.device != 'cpu' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        args.device = 'cpu'
    
    # Train model
    model, results = train_yolov8(
        data_config=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        save_dir=args.save_dir,
        project_name=args.project
    )
    
    # Evaluate model if requested
    if args.evaluate:
        evaluate_model(model, args.data, results.save_dir)
    
    # Save sample predictions
    save_sample_predictions(model, args.data, results.save_dir, args.samples)
    
    print(f"\nTraining pipeline completed!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Model weights: {results.save_dir / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main() 