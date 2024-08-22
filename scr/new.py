from ultralytics import YOLO
model = YOLO(r"C:\Users\USER\Documents\Drowsiness Guard\BSafe\best.pt")
model.predict(source=0, show=True)
#python chat_gpt.py model=best.pt source="test3.mp4" show=True
