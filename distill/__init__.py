"""
Knowledge Distillation Module for YOLOv8
Contains ensemble teacher and distillation utilities.
"""

from .ensemble_teacher import EnsembleTeacher, create_ensemble_teacher

__all__ = ['EnsembleTeacher', 'create_ensemble_teacher'] 