from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from ultralytics import YOLO
import math
from typing import List
import time

app = FastAPI(
    title="Drowsy Guard Detection API",
    description="""Obtain drowsiness state value out of and image/ video
    and return image and json result""",
    version="0.0.1",
)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"]
app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],)

model = YOLO("Model/best.pt")

# List of class names (adjust according to your model's classes)
classNames = ['Eye_Closed', 'Eye_Open', 'Facing_Front', 'Mouth_Yawning']

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)
    
# Endpoint for image upload and processing
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    nparr = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run the model on the image
    results = model(img, stream=True)
    
    detected_classes = []

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Extract and format confidence and class label
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            print("Here:", cls)
            if cls > len(classNames)-1:
                print(f"cls: {cls}, len(classNames): {len(classNames)}")
                continue
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            detected_classes.append({"class_name": class_name, "confidence": conf})
            
            # Calculate label size
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            
            # Draw filled rectangle for the label background
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            
            # Put label text on the image
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # Encode the processed image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    # Return the detected classes and image
    return JSONResponse(content={
        "detected_classes": detected_classes,
    })

@app.post("/detect_and_image/")
async def detect_image_and_return(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    nparr = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run the model on the image
    results = model(img, stream=True)
    
    detected_classes = []

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Extract and format confidence and class label
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls > len(classNames)-1:
                print(f"cls: {cls}, len(classNames): {len(classNames)}")
                continue
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            detected_classes.append({"class_name": class_name, "confidence": conf})
            
            # Calculate label size
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            
            # Draw filled rectangle for the label background
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            
            # Put label text on the image
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # Encode the processed image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    return StreamingResponse(img_bytes, media_type="image/jpeg")
# Live video stream route

@app.get("/video_stream/")
async def video_stream():
    def generate():
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

        while True:
            success, img = cap.read()
            if not success:
                break
            
            # Run the model on the frame
            results = model(img, stream=True)
            
            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Extract and format confidence and class label
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if cls > len(classNames)-1:
                        print(f"cls: {cls}, len(classNames): {len(classNames)}")
                        continue
                    class_name = classNames[cls]
                    label = f'{class_name} {conf}'
                    
                    # Calculate label size
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    
                    # Draw filled rectangle for the label background
                    cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                    
                    # Put label text on the image
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            
            # Yield the frame as part of a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")



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
            
            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Extract and format confidence and class label
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if cls > len(classNames)-1:
                        print(f"cls: {cls}, len(classNames): {len(classNames)}")
                        continue
                    detected_class = classNames[cls]
                    # Send detected class as JSON
                    yield f"data: {detected_class}\n\n"
            time.sleep(1)  # Optional: Adjust the sleep time as needed
        cap.release()
    return StreamingResponse(event_stream(), media_type="text/event-stream")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
