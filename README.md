# 🤟 ASL Detection - Sign Language Recognition System

A real-time American Sign Language (ASL) detection system using YOLOv8 for recognizing hand gestures and building sentences from sign language.

## 📋 Features

- **Real-time ASL Detection**: Recognizes 26 letters (A-Z) + backspace gesture
- **Sentence Builder**: Automatically builds sentences from detected gestures
- **Custom Dataset Support**: Merge multiple datasets for improved accuracy
- **YOLOv8 Integration**: State-of-the-art object detection model
- **GPU Acceleration**: CUDA support for faster inference and training

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam for real-time detection
- (Optional) NVIDIA GPU with CUDA for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PrajanManojKumarRekha/Object-Detection.git
   cd Object-Detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Usage Guide

### Step 1: Prepare Your Dataset (Optional)

If you have multiple ASL datasets to merge:

```bash
python merge_datasets.py
```

This will:
- Combine datasets from different sources
- Create a unified `ASL_Merged` directory
- Generate a `data.yaml` configuration file

### Step 2: Train the Model

Train the YOLOv8 model on your ASL dataset:

```bash
python train_model.py
```

**Training Configuration:**
- Model: YOLOv8 Nano (yolov8n.pt)
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640
- Early Stopping: 10 epochs patience

**What happens during training:**
1. Pre-training checks (dataset validation, GPU detection)
2. Model training with automatic checkpointing
3. Validation on test set
4. Best model saved to `models/best_asl_27.pt`

**Expected Output:**
```
✓ GPU Available: [Your GPU Name]
✓ Merged dataset found
✓ Model: yolov8n.pt
✓ Epochs: 100
✓ Batch Size: 16

📊 Validation Results:
  mAP@0.5:    0.XXX
  mAP@0.5-95: 0.XXX
  Precision:  0.XXX
  Recall:     0.XXX
```

### Step 3: Run Real-Time Detection

Start the ASL detection system:

```bash
python Runner.py
```

**Controls:**
- **Q**: Quit the application
- **C**: Clear the sentence builder
- **S**: Show statistics (if implemented)

**How it works:**
1. Opens your webcam
2. Detects ASL gestures in real-time
3. Displays bounding boxes with confidence scores
4. Builds sentences from detected letters
5. Shows FPS and current sentence at the bottom

## 📁 Project Structure

```
Object-Detection/
├── ASL_Merged/              # Merged dataset directory
│   ├── train/              # Training images
│   ├── valid/              # Validation images
│   ├── test/               # Test images
│   └── data.yaml           # Dataset configuration
├── models/                  # Trained model weights
│   └── best_asl_27.pt      # Best trained model
├── runs/                    # Training runs and results
├── OpenCV collector/        # Data collection scripts
├── merge_datasets.py        # Dataset merging utility
├── train_model.py          # Model training script
├── Runner.py               # Real-time inference script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Recognized Gestures

The system recognizes 27 different gestures:
- **A-Z**: All 26 letters of the alphabet
- **Backspace**: Delete the last character

## ⚙️ Configuration

### Modify Training Parameters

Edit `train_model.py`:
```python
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 16          # Batch size (reduce if out of memory)
IMAGE_SIZE = 640         # Input image size
PATIENCE = 10            # Early stopping patience
```

### Modify Inference Settings

Edit `Runner.py`:
```python
CONFIDENCE_THRESHOLD = 0.6    # Minimum confidence for detection
DEBOUNCE_TIME = 1.5          # Time between same gesture detection (seconds)
```

## 🔧 Troubleshooting

### Issue: "Merged dataset not found"
**Solution:** Run `python merge_datasets.py` first to create the merged dataset.

### Issue: "Cannot access webcam"
**Solution:** 
- Check if another application is using the webcam
- Verify webcam permissions in your OS settings
- Try changing camera index in `Runner.py`: `cv2.VideoCapture(1)` instead of `0`

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Issue: Low detection accuracy
**Solution:**
- Train for more epochs
- Add more training data
- Adjust confidence threshold in `Runner.py`
- Ensure good lighting conditions during inference

## 📊 Performance Tips

1. **GPU Training**: Use CUDA-enabled GPU for 10-20x faster training
2. **Lighting**: Ensure good, consistent lighting for better detection
3. **Background**: Use a plain background for improved accuracy
4. **Distance**: Keep hand at consistent distance from camera
5. **Gesture Hold**: Hold each gesture steady for 1-2 seconds

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: For the excellent object detection framework
- **OpenCV**: For computer vision capabilities
- **ASL Community**: For gesture datasets and resources

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the ASL community**
