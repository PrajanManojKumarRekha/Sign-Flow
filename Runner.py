"""
Real-time ASL Inference Script
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

# Configuration
MODEL_PATH = "models/best_asl_27.pt"
CONFIDENCE_THRESHOLD = 0.6
DEBOUNCE_TIME = 1.5

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'backspace'
]

class SentenceBuilder:
    def __init__(self):
        self.text = ""
        self.last_gesture = None
        self.last_time = 0
        
    def add_character(self, char, current_time):
        if char == self.last_gesture and (current_time - self.last_time) < DEBOUNCE_TIME:
            return False
        
        if char == 'backspace':
            if self.text:
                self.text = self.text[:-1]
                print(f"[BACKSPACE] → '{self.text}'")
        else:
            self.text += char
            print(f"[+{char}] → '{self.text}'")
        
        self.last_gesture = char
        self.last_time = current_time
        return True
    
    def get_text(self):
        return self.text if self.text else "[Empty]"
    
    def clear(self):
        self.text = ""
        print("[CLEARED]")

def main():
    print("\n" + "="*60)
    print("ASL REAL-TIME DETECTION")
    print("="*60)
    
    print(f"\nLoading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nRun: python train_model.py first")
        return
    
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Webcam ready")
    
    builder = SentenceBuilder()
    
    print("\n" + "="*60)
    print("CONTROLS: Q=Quit | C=Clear | S=Stats")
    print("="*60 + "\n")
    
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES[class_id]
                
                color = (0, 165, 255) if class_name == 'backspace' else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    builder.add_character(class_name, current_time)
        
        fps_counter += 1
        if current_time - fps_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_time = current_time
        
        panel_height = 180
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        cv2.putText(panel, "ASL Sentence Builder", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        text_display = builder.get_text()
        if len(text_display) > 50:
            text_display = "..." + text_display[-47:]
        
        cv2.rectangle(panel, (10, 45), (frame.shape[1] - 10, 100), (60, 60, 60), -1)
        cv2.putText(panel, text_display, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(panel, "Q: Quit  |  C: Clear", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(frame, f"FPS: {fps}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        combined = np.vstack([frame, panel])
        cv2.imshow("ASL Detection", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            builder.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Final text: {builder.text if builder.text else '[Empty]'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()