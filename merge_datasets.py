"""
Dataset Merger Script
Combines A-Z and Backspace datasets into a single 27-class dataset
"""

import os
import shutil
import yaml
from pathlib import Path

# Configuration
DATASET_A_Z = "ASL.v1i.yolov8"
DATASET_BACKSPACE = "ASL-Custom-Gestures-1"
MERGED_DATASET_DIR = "ASL_Merged"

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'backspace'
]

class DatasetMerger:
    def __init__(self):
        self.merged_dir = MERGED_DATASET_DIR
        
    def create_directory_structure(self):
        print("\n" + "="*60)
        print("Creating Directory Structure")
        print("="*60)
        
        for split in ['train', 'valid', 'test']:
            os.makedirs(f"{self.merged_dir}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.merged_dir}/{split}/labels", exist_ok=True)
        
        print(f"✓ Created directory: {os.path.abspath(self.merged_dir)}")
        
    def copy_dataset(self, source_dir, class_offset=0, dataset_name=""):
        print(f"\n📁 Copying {dataset_name} dataset...")
        stats = {'train': 0, 'valid': 0, 'test': 0}
        
        for split in ['train', 'valid', 'test']:
            src_img_dir = f"{source_dir}/{split}/images"
            src_lbl_dir = f"{source_dir}/{split}/labels"
            dst_img_dir = f"{self.merged_dir}/{split}/images"
            dst_lbl_dir = f"{self.merged_dir}/{split}/labels"
            
            if not os.path.exists(src_img_dir):
                print(f"  ⚠ Warning: {src_img_dir} not found, skipping...")
                continue
            
            # Copy images
            for img_file in os.listdir(src_img_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    src_path = f"{src_img_dir}/{img_file}"
                    dst_path = f"{dst_img_dir}/{img_file}"
                    
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(img_file)
                        dst_path = f"{dst_img_dir}/{dataset_name}_{img_file}"
                    
                    shutil.copy2(src_path, dst_path)
                    stats[split] += 1
            
            # Copy and adjust labels
            if os.path.exists(src_lbl_dir):
                for lbl_file in os.listdir(src_lbl_dir):
                    if lbl_file.endswith('.txt'):
                        src_path = f"{src_lbl_dir}/{lbl_file}"
                        dst_path = f"{dst_lbl_dir}/{lbl_file}"
                        
                        if os.path.exists(dst_path):
                            base, ext = os.path.splitext(lbl_file)
                            dst_path = f"{dst_lbl_dir}/{dataset_name}_{lbl_file}"
                        
                        if class_offset == 0:
                            shutil.copy2(src_path, dst_path)
                        else:
                            with open(src_path, 'r') as f:
                                lines = f.readlines()
                            
                            with open(dst_path, 'w') as f:
                                for line in lines:
                                    parts = line.strip().split()
                                    if parts:
                                        class_id = int(parts[0]) + class_offset
                                        parts[0] = str(class_id)
                                        f.write(' '.join(parts) + '\n')
        
        print(f"  ✓ Train: {stats['train']} | Valid: {stats['valid']} | Test: {stats['test']}")
        return stats
    
    def create_data_yaml(self):
        print("\n" + "="*60)
        print("Creating data.yaml")
        print("="*60)
        
        data_yaml = {
            'path': os.path.abspath(self.merged_dir),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(CLASS_NAMES),
            'names': CLASS_NAMES
        }
        
        yaml_path = f"{self.merged_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✓ Created: {os.path.abspath(yaml_path)}")
        print(f"Total Classes: {len(CLASS_NAMES)}")
        
    def merge(self):
        print("\n" + "="*60)
        print("ASL DATASET MERGER")
        print("="*60)
        
        self.create_directory_structure()
        
        print("\n" + "="*60)
        print("Merging Datasets")
        print("="*60)
        
        stats_az = self.copy_dataset(DATASET_A_Z, class_offset=0, dataset_name="AZ")
        stats_back = self.copy_dataset(DATASET_BACKSPACE, class_offset=26, dataset_name="backspace")
        
        total_train = stats_az['train'] + stats_back['train']
        total_valid = stats_az['valid'] + stats_back['valid']
        total_test = stats_az['test'] + stats_back['test']
        
        print(f"\n📈 Total Images:")
        print(f"  Train: {total_train} | Valid: {total_valid} | Test: {total_test}")
        
        self.create_data_yaml()
        
        print("\n" + "="*60)
        print("✅ DATASET MERGE COMPLETE!")
        print("="*60)

if __name__ == "__main__":
    merger = DatasetMerger()
    merger.merge()