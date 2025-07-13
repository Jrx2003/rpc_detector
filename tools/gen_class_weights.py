#!/usr/bin/env python3
"""
Generate class weights for RPC dataset to mitigate long-tail class imbalance.
Computes weights based on class frequency using the formula: weight = (median(freq) / freq_i)**0.5
"""

import os
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from tqdm import tqdm

def count_class_instances(labels_dir):
    """
    Count instances per class in YOLO label files.
    
    Args:
        labels_dir: Directory containing YOLO label files
        
    Returns:
        dict: Class ID to count mapping
    """
    class_counts = defaultdict(int)
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"Found {len(label_files)} label files")
    
    for label_file in tqdm(label_files, desc="Counting class instances"):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    return class_counts

def compute_class_weights(class_counts, num_classes=200):
    """
    Compute class weights using the formula: weight = (median(freq) / freq_i)**0.5
    
    Args:
        class_counts: Dict mapping class ID to count
        num_classes: Total number of classes
        
    Returns:
        numpy.ndarray: Class weights array
    """
    # Create frequency array for all classes
    frequencies = np.zeros(num_classes)
    
    for class_id, count in class_counts.items():
        if class_id < num_classes:  # Ensure class_id is within bounds
            frequencies[class_id] = count
    
    # Handle classes with zero instances (set to 1 to avoid division by zero)
    frequencies[frequencies == 0] = 1
    
    # Calculate median frequency
    median_freq = np.median(frequencies)
    
    # Compute weights: (median(freq) / freq_i)**0.5
    weights = (median_freq / frequencies) ** 0.5
    
    print(f"Class frequency statistics:")
    print(f"  Min frequency: {frequencies.min()}")
    print(f"  Max frequency: {frequencies.max()}")
    print(f"  Median frequency: {median_freq}")
    print(f"  Mean frequency: {frequencies.mean():.2f}")
    
    print(f"\nClass weight statistics:")
    print(f"  Min weight: {weights.min():.4f}")
    print(f"  Max weight: {weights.max():.4f}")
    print(f"  Mean weight: {weights.mean():.4f}")
    
    return weights

def save_class_weights(weights, output_path):
    """
    Save class weights to numpy file.
    
    Args:
        weights: Class weights array
        output_path: Path to save the weights file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, weights)
    print(f"Class weights saved to: {output_path}")

def generate_class_weights(labels_dir='data/RPC/train/labels', 
                          output_path='data/RPC/class_weights.npy',
                          num_classes=200):
    """
    Generate class weights for the RPC dataset.
    
    Args:
        labels_dir: Directory containing training label files
        output_path: Path to save the class weights
        num_classes: Total number of classes
    """
    print("Generating class weights for RPC dataset...")
    print(f"Labels directory: {labels_dir}")
    print(f"Output path: {output_path}")
    print(f"Number of classes: {num_classes}")
    
    # Count class instances
    class_counts = count_class_instances(labels_dir)
    
    # Compute class weights
    weights = compute_class_weights(class_counts, num_classes)
    
    # Save weights
    save_class_weights(weights, output_path)
    
    # Print some statistics
    print(f"\nClass distribution analysis:")
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 most frequent classes:")
    for class_id, count in sorted_counts[:10]:
        print(f"  Class {class_id}: {count} instances (weight: {weights[class_id]:.4f})")
    
    print(f"\nTop 10 least frequent classes:")
    for class_id, count in sorted_counts[-10:]:
        print(f"  Class {class_id}: {count} instances (weight: {weights[class_id]:.4f})")
    
    return weights

def main():
    parser = argparse.ArgumentParser(description='Generate class weights for RPC dataset')
    parser.add_argument('--labels-dir', default='data/RPC/train/labels', 
                       help='Directory containing training label files')
    parser.add_argument('--output', default='data/RPC/class_weights.npy',
                       help='Output path for class weights file')
    parser.add_argument('--num-classes', type=int, default=200,
                       help='Total number of classes')
    
    args = parser.parse_args()
    
    # Generate class weights
    weights = generate_class_weights(
        labels_dir=args.labels_dir,
        output_path=args.output,
        num_classes=args.num_classes
    )
    
    print(f"\nClass weights generation completed!")
    print(f"Use these weights in training by setting: cls_pw: {args.output}")

if __name__ == "__main__":
    main() 