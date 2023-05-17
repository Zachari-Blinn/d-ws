from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, IndexModel, ASCENDING
from ultralytics import YOLO
import cv2
import os
import supervision as sv
import time

load_dotenv()

# Constants
CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = "yolov8n.pt"
SOURCE = 0
MONGO_URI = "mongodb://root:example@localhost:27017/"
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
TIME_BETWEEN_SAVES = 5  # seconds
TIME_TO_LIVE_DATA = 2*24*60*60  # seconds

# Database setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
objects_collection = db[COLLECTION_NAME]
# Create a TTL index on the "datetime" field with an expiration of 2 days
index = IndexModel([("datetime", ASCENDING)], expireAfterSeconds=TIME_TO_LIVE_DATA)
objects_collection.create_indexes([index])

def process_frame(frame, model, box_annotator, saved_objects, last_saved_time):
  results = model.track(frame)[0]
  
  detections = sv.Detections(
    xyxy=results.boxes.xyxy.cpu().numpy(),
    confidence=results.boxes.conf.cpu().numpy(),
    class_id=results.boxes.cls.cpu().numpy().astype(int),
  )
  
  detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
  
  if results.boxes.id is not None:
    detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
  
  labels = [
    f"#{tracker_id} {model.model.names[class_id]} {confidence:.2f}"
    for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id if detections.tracker_id is not None else [None] * len(detections.class_id))
  ]
  
  annotated_frame = box_annotator.annotate(
    scene=frame,
    detections=detections,
    labels=labels,
  )
          
  # Save to DB every 5 seconds or if an object has not been seen for 60 seconds
  current_time = time.time()
  if current_time - last_saved_time >= TIME_BETWEEN_SAVES or (detections.tracker_id is not None and any(current_time - saved_objects.get(tracker_id, 0) >= 60 for tracker_id in detections.tracker_id)):
    for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id if detections.tracker_id is not None else [None] * len(detections.class_id)):
      doc = {
        "datetime": str(datetime.now()),
        "detected_object_name": str(model.model.names[class_id]),
        "detected_object_id": str(tracker_id),
        "confidence": str(confidence),
      }
      objects_collection.insert_one(doc)
      saved_objects[tracker_id] = current_time
      print("SAVED TO DB: ", doc)
      # print first 10 rows of collection
      print(list(objects_collection.find().limit(10)))
    last_saved_time = current_time

  return annotated_frame, last_saved_time

def main():
  model = YOLO(MODEL_PATH)
  camera = cv2.VideoCapture(SOURCE)
  if not camera.isOpened():
    raise RuntimeError("Unable to open camera")
  
  print("Starting camera...\nPress 'q' to quit.")
  
  box_annotator = sv.BoxAnnotator()
  saved_objects = {}  # dictionary of saved object ids and their last seen time
  last_saved_time = time.time()

  while camera.isOpened():
    success, frame = camera.read()
    if not success:
      break
      
    annotated_frame, last_saved_time = process_frame(frame, model, box_annotator, saved_objects, last_saved_time)
    
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  camera.release()
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
