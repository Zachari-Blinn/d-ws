from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

out = model.predict("/home/zachari-blinn/Projects/other/d-ws/object-detection-module/data/test/living_room.jpg", conf=0.6)

print(out)

out.show()