from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load your trained YOLO model
import os
# Load your trained YOLO model
model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
model = YOLO(model_path)

# Automatically get class names from model
vegetable_classes = list(model.names.values())
print("Vegetable classes detected from model:", vegetable_classes)

# For counting each vegetable type
counts = defaultdict(int)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True, conf=0.5)  # confidence threshold

    # Reset counts for each frame
    frame_counts = defaultdict(int)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            # Count only detected vegetable classes
            if cls_name in vegetable_classes:
                frame_counts[cls_name] += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    # Show counts in a fancy overlay box
    overlay = frame.copy()
    box_height = 25 * (len(frame_counts) + 1)
    cv2.rectangle(overlay, (10, 10), (250, 10 + box_height), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    y_offset = 35
    cv2.putText(frame, "Vegetable Counts:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    for veg, count in frame_counts.items():
        cv2.putText(frame, f"{veg}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    # Show the frame
    cv2.imshow("Vegetable Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release("q")
cv2.destroyAllWindows()
