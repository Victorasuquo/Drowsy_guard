
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import time

app = FastAPI()

# Dummy class names (replace with actual classes)
classNames = ['Eye_Closed', 'Eye_Open', 'Facing_Front', 'Mouth_Yawning']

# Replace with your model initialization
model = YOLO("model.pt") # Load your model here

def generate_video_feed():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Run the model on the frame
        results = model(img, stream=True)
        
        detected_class = None

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract class ID
                cls = int(box.cls[0])
                print("Here:", cls)
                if cls > len(classNames)-1:
                    print(f"cls: {cls}, len(classNames): {len(classNames)}")
                    continue
                detected_class = classNames[cls]
                
                # Draw bounding box (optional)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.get("/video_stream/")
async def video_stream():
    return StreamingResponse(generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detected_class_stream/")
async def detected_class_stream():
    async def event_stream():
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

        while True:
            success, img = cap.read()
            if not success:
                break

            # Run the model on the frame
            results = model(img, stream=True)
            
            detected_class = None

            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract class ID
                    cls = int(box.cls[0])
                    print("Here:", cls)
                    if cls > len(classNames)-1:
                        print(f"cls: {cls}, len(classNames): {len(classNames)}")
                        continue
                    detected_class = classNames[cls]
                    
                    # Send detected class as JSON
                    yield f"data: {detected_class}\n\n"

            time.sleep(1)  # Optional: Adjust the sleep time as needed

        cap.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
