from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from ultralytics import YOLO
import cv2
import os

load_dotenv()

CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = "yolov8n.pt"
SOURCE = "0"
MONGO_URI = "mongodb://root:example@localhost:27017/"
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
objects_collection = db[COLLECTION_NAME]

def main():
  model = YOLO(MODEL_PATH)
  camera = cv2.VideoCapture(0)
  
  while camera.isOpened():
    success, frame = camera.read()
    
    if success:
      results = model(frame)
      
      cls = results[0].boxes.cls
      print(cls)

      annotated_frame = results[0].plot()

      cv2.imshow("YOLOv8 Inference", annotated_frame)
      
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else:
      break
  
  camera.release()
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
