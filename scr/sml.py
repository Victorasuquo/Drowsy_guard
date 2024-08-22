from ultralytics import YOLO
import cv2
import math

# Initialize video capture with the default webcam
cap = cv2.VideoCapture(0)

# Check if the video capture was successfully initialized
if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

# Define the frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Load the YOLO model
model = YOLO("best.pt")

# Define the class names based on your model's training
classNames = ['Eye_Closed', 'Eye_Open', 'Facing_Front', 'Mouth_Yawning']

while True:
    # Capture a frame from the webcam
    success, img = cap.read()
    
    if not success:
        print("Error: Failed to read frame from video or webcam.")
        break

    # Perform detection on the frame
    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = box.xyxy[0].int()
            print(f"Box coordinates: {x1, y1, x2, y2}")
            
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Extract confidence and class label
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            
            # Calculate the size of the label
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            
            # Draw a filled rectangle for the label background
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            
            # Put the label text on the image
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Display the frame with detections
    cv2.imshow("Live Detection", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
