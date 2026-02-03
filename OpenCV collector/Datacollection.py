import cv2
import os

CLASS_NAME = "next_line"
SAVE_DIR = f"custom_data/{CLASS_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (200, 100), (450, 350), (0,255,0), 2)
    cv2.putText(frame, f"Images: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):  # press C to capture
        cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", frame)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()