@echo off
echo Starting baseline training...
python train.py --epochs 10 --batch 16 --imgsz 640 --evaluate

echo Starting training with focal loss...
python train_focal.py --epochs 10 --batch 16 --imgsz 640 --evaluate

echo Starting training with distillation...
python train_distill.py --epochs 10 --batch 16 --imgsz 640 --evaluate

echo Evaluation complete.
pause