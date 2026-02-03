"""
YOLOv8 Training Script
Trains the ASL detection model on merged dataset
"""

import os
import torch
from ultralytics import YOLO
from datetime import datetime

# Configuration
MERGED_DATASET_DIR = "ASL_Merged"
MODEL_SIZE = "yolov8n.pt"
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "training_results"
MODEL_SAVE_PATH = "models"

class ASLTrainer:
    def __init__(self):
        self.model = None
        self.results = None
        self.data_yaml = f"{MERGED_DATASET_DIR}/data.yaml"
        
    def check_setup(self):
        print("\n" + "="*60)
        print("PRE-TRAINING CHECKS")
        print("="*60)
        
        if not os.path.exists(MERGED_DATASET_DIR):
            print("❌ Merged dataset not found!")
            print("   Please run: python merge_datasets.py first")
            return False
        
        if not os.path.exists(self.data_yaml):
            print("❌ data.yaml not found!")
            return False
        
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Available: {gpu_name}")
        else:
            print("⚠ GPU not detected - using CPU (slower)")
        
        print(f"✓ Merged dataset found")
        print(f"✓ Model: {MODEL_SIZE}")
        print(f"✓ Epochs: {EPOCHS}")
        print(f"✓ Batch Size: {BATCH_SIZE}")
        
        return True
    
    def load_model(self):
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)
        
        print(f"Loading {MODEL_SIZE}...")
        self.model = YOLO(MODEL_SIZE)
        print(f"✓ Model loaded\n")
    
    def train(self):
        print("="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        self.results = self.model.train(
            data=self.data_yaml,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            name='asl_27_classes',
            patience=PATIENCE,
            save=True,
            project=OUTPUT_DIR,
            device=DEVICE,
            verbose=True,
            plots=True
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
    
    def validate(self):
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)
        
        metrics = self.model.val()
        
        print(f"\n📊 Validation Results:")
        print(f"  mAP@0.5:    {metrics.box.map50:.3f}")
        print(f"  mAP@0.5-95: {metrics.box.map:.3f}")
        print(f"  Precision:  {metrics.box.mp:.3f}")
        print(f"  Recall:     {metrics.box.mr:.3f}")
        
        if metrics.box.map50 >= 0.85:
            print("\n✅ Excellent performance!")
        elif metrics.box.map50 >= 0.70:
            print("\n✓ Good performance")
        else:
            print("\n⚠ Consider more data or training")
    
    def save_final_model(self):
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        import shutil
        
        best_model_src = f"{OUTPUT_DIR}/asl_27_classes/weights/best.pt"
        best_model_dst = f"{MODEL_SAVE_PATH}/best_asl_27.pt"
        
        if os.path.exists(best_model_src):
            shutil.copy(best_model_src, best_model_dst)
            print(f"✓ Model saved to: {os.path.abspath(best_model_dst)}")
        
    def run(self):
        if not self.check_setup():
            return False
        
        self.load_model()
        self.train()
        self.validate()
        self.save_final_model()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print("\nNext: python inference.py\n")
        
        return True

if __name__ == "__main__":
    trainer = ASLTrainer()
    trainer.run()