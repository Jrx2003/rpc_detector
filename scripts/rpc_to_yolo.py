#!/usr/bin/env python3
"""
RPC Dataset to YOLO Format Converter
Converts RPC dataset from COCO format to YOLO format with train/val/test splits.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import yaml

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox format to YOLO format.
    
    Args:
        coco_bbox: [x, y, width, height] in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = coco_bbox
    
    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    return [x_center, y_center, w, h]

def create_yolo_dataset(root_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Convert RPC dataset to YOLO format.
    
    Args:
        root_dir: Path to RPC dataset root directory
        output_dir: Output directory for YOLO format dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Paths
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    annotations_file = root_dir / "instances_test2019.json"
    images_dir = root_dir / "test2019"
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping (COCO uses 1-based indexing, YOLO uses 0-based)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create image info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Get all image IDs and shuffle for random split
    image_ids = list(image_info.keys())
    random.shuffle(image_ids)
    
    # Split dataset
    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Process each split
    for split_name, split_ids in splits.items():
        print(f"\nProcessing {split_name} set...")
        
        for image_id in tqdm(split_ids, desc=f"Converting {split_name}"):
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img_path = images_dir / img_filename
            dst_img_path = output_dir / split_name / 'images' / img_filename
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
                
                # Create label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = output_dir / split_name / 'labels' / label_filename
                
                with open(label_path, 'w') as f:
                    if image_id in image_annotations:
                        for ann in image_annotations[image_id]:
                            # Convert bbox to YOLO format
                            yolo_bbox = coco_to_yolo_bbox(ann['bbox'], img_width, img_height)
                            
                            # Convert category ID to YOLO format (0-based)
                            yolo_class_id = category_id_to_yolo[ann['category_id']]
                            
                            # Write to file: class_id x_center y_center width height
                            f.write(f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n")
            else:
                print(f"Warning: Image {img_filename} not found")
    
    # Create YOLO dataset configuration file
    create_yolo_config(output_dir, categories, category_id_to_yolo)
    
    print(f"\nDataset conversion complete!")
    print(f"Output directory: {output_dir}")

def create_yolo_config(output_dir, categories, category_id_to_yolo):
    """Create YOLO dataset configuration file."""
    
    # Create class names list in YOLO order
    class_names = [''] * len(categories)
    for coco_id, yolo_id in category_id_to_yolo.items():
        class_names[yolo_id] = categories[coco_id]
    
    # Create YAML configuration
    config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(categories),
        'names': class_names
    }
    
    # Save configuration file
    config_path = output_dir / 'rpc.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YOLO config file: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert RPC dataset to YOLO format')
    parser.add_argument('--root', required=True, help='Path to RPC dataset root directory')
    parser.add_argument('--output', default='data/RPC', help='Output directory for YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Convert dataset
    create_yolo_dataset(
        root_dir=args.root,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

if __name__ == "__main__":
    main() 