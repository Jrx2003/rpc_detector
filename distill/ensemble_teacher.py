#!/usr/bin/env python3
"""
Ensemble Teacher Module for Knowledge Distillation
Combines YOLOv8x and YOLOv8l models and averages their outputs during forward pass.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np


class EnsembleTeacher(nn.Module):
    """
    Ensemble teacher model that combines YOLOv8x and YOLOv8l models.
    Averages their logits and bbox regression outputs during forward pass.
    """
    
    def __init__(self, teacher_models=['yolov8x.pt', 'yolov8l.pt']):
        """
        Initialize the ensemble teacher.
        
        Args:
            teacher_models: List of teacher model paths/names
        """
        super(EnsembleTeacher, self).__init__()
        
        self.teacher_models = []
        print(f"Loading teacher models: {teacher_models}")
        
        for model_path in teacher_models:
            print(f"Loading {model_path}...")
            model = YOLO(model_path)
            
            # Set to evaluation mode and freeze parameters
            model.model.eval()
            for param in model.model.parameters():
                param.requires_grad = False
            
            self.teacher_models.append(model.model)
            print(f"Successfully loaded {model_path}")
        
        # Store number of teachers
        self.num_teachers = len(self.teacher_models)
        print(f"Ensemble teacher initialized with {self.num_teachers} models")
    
    def forward(self, x):
        """
        Forward pass through ensemble teacher.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged outputs from all teacher models
        """
        with torch.no_grad():
            outputs = []
            
            for teacher in self.teacher_models:
                teacher_out = teacher(x)
                outputs.append(teacher_out)
            
            # Average the outputs
            if len(outputs) > 1:
                # Handle different output formats
                if isinstance(outputs[0], (list, tuple)):
                    # Multi-scale outputs
                    averaged_outputs = []
                    for i in range(len(outputs[0])):
                        scale_outputs = [out[i] for out in outputs]
                        averaged_scale = torch.stack(scale_outputs).mean(dim=0)
                        averaged_outputs.append(averaged_scale)
                    return averaged_outputs
                else:
                    # Single output
                    return torch.stack(outputs).mean(dim=0)
            else:
                return outputs[0]
    
    def get_teacher_predictions(self, x):
        """
        Get predictions from individual teachers for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of predictions from each teacher
        """
        predictions = []
        
        with torch.no_grad():
            for teacher in self.teacher_models:
                pred = teacher(x)
                predictions.append(pred)
        
        return predictions
    
    def extract_features(self, x, teacher_idx=None):
        """
        Extract features from specific teacher or ensemble.
        
        Args:
            x: Input tensor
            teacher_idx: Index of specific teacher (None for ensemble)
            
        Returns:
            Feature representations
        """
        if teacher_idx is not None:
            with torch.no_grad():
                return self.teacher_models[teacher_idx](x)
        else:
            return self.forward(x)


def create_ensemble_teacher(teacher_weights=['yolov8x.pt', 'yolov8l.pt']):
    """
    Factory function to create ensemble teacher.
    
    Args:
        teacher_weights: List of teacher model weights
        
    Returns:
        EnsembleTeacher instance
    """
    return EnsembleTeacher(teacher_weights)


if __name__ == "__main__":
    # Test the ensemble teacher
    print("Testing Ensemble Teacher...")
    
    # Create ensemble teacher
    teacher = create_ensemble_teacher()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    outputs = teacher(dummy_input)
    
    if isinstance(outputs, (list, tuple)):
        print(f"Ensemble outputs: {len(outputs)} scales")
        for i, output in enumerate(outputs):
            print(f"  Scale {i}: {output.shape}")
    else:
        print(f"Ensemble output shape: {outputs.shape}")
    
    print("Ensemble Teacher test completed successfully!") 