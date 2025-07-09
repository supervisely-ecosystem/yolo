import os
from dotenv import load_dotenv
import supervisely as sly

from supervisely_integration.serve.serve_yolo import YOLOModel

if sly.is_development():
    load_dotenv("local.env")
    # load_dotenv("supervisely.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model_n = 1

# 1. Pretrained model
if model_n == 1:
    model = YOLOModel(
        model="YOLO11n-det",
        device="cuda",
    )

# 2. Local checkpoint
elif model_n == 2:
    model = YOLOModel(
        model="my_models/best.pth",
        device="cuda",
    )

# 3. Remote Custom Checkpoint (Team Files)
elif model_n == 3:
    model = YOLOModel(
        model="/experiments/9_Animals (Bitmap)/47698_YOLO/checkpoints/best.pt",
        device="cuda:0",
    )

image_path = "supervisely_integration/demo/img/coco_sample.jpg"
predictions = model(input=image_path)
print(f"Predictions: {len(predictions)}")
