from ultralytics import YOLO
import cv2
import math


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"Error: Could not open video or webcam at path PATH")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
model = YOLO("best.pt")
classNames = ['Eye_Closed', 'Eye_Open', 'Facing_Front', 'Mouth_Yawning']

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from video or webcam.")
        break

    results = model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)  # Ensure coordinates are integers
            print(f"Box coordinates: {x1, y1, x2, y2}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow(img)

cap.release()
cv2.destroyAllWindows()