from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from ultralytics import YOLO
import cv2
import os
import supervision as sv

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
  CLASS_NAMES_DICT = model.model.names
  camera = cv2.VideoCapture(0)
  print("Starting camera...")
  print('Press "q" to quit')
  print("CLASS_NAMES_DICT: ", CLASS_NAMES_DICT)
  while camera.isOpened():
    success, frame = camera.read()
    
    if success:
      results = model(frame)[0]
      
      detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int),
      )
      
      detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
      
      box_annotator = sv.BoxAnnotator()

      labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
      ]
      
      annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels,
      )
      
      cv2.imshow("YOLOv8 Inference", annotated_frame)
      
      # for label in labels:
      #   print("LABELS: ",label)
      
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else:
      break
  
  camera.release()
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
