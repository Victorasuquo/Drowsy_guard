from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
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

def load_image(file: UploadFile) -> np.ndarray:
    image_bytes = file.read()
    nparr = np.fromstring(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def run_model(image: np.ndarray) -> List[dict]:
    results = model(image, stream=True)
    detected_classes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls > len(classNames)-1:
                continue
            class_name = classNames[cls]
            detected_classes.append({"class_name": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})
    return detected_classes

def draw_bounding_boxes(image: np.ndarray, detected_classes: List[dict]) -> np.ndarray:
    for detection in detected_classes:
        class_name, conf = detection["class_name"], detection["confidence"]
        label = f'{class_name} {conf}'
        x1, y1, x2, y2 = [int(x) for x in detection["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return image

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image = load_image(file)
    detected_classes = run_model(image)
    image = draw_bounding_boxes(image, detected_classes)
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    return JSONResponse(content={"detected_classes": detected_classes})

@app.post("/detect_and_image/")
async def detect_image_and_return(file: UploadFile = File(...)):
    image = load_image(file)
    detected_classes = run_model(image)
    image = draw_bounding_boxes(image, detected_classes)
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    return StreamingResponse(img_bytes, media_type="image/jpeg")

@app.get("/video_stream/")
async def video_stream():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                break
            detected_classes = run_model(img)
            img = draw_bounding_boxes(img, detected_classes)
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detected_class_stream/")
async def detected_class_stream():
    async def event_stream():
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                break
            detected_classes = run_model(img)
            for detection in detected_classes:
                yield f"data: {detection['class_name']}\n\n"
            time.sleep(1)  # Optional: Adjust the sleep time as needed
        cap.release()
    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)