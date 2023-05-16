from pymongo import MongoClient
from ultralytics import YOLO
from datetime import datetime
import os
from dotenv import load_dotenv

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
  results = model.predict(source=SOURCE, show=True, conf=CONFIDENCE_THRESHOLD, stream=True)
  for result in results:
    print(result)
      
if __name__ == "__main__":
  main()
