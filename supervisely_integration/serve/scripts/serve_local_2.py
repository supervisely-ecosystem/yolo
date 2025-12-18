import os
from pathlib import Path

import supervisely as sly
from supervisely.nn.model.model_api import ModelAPI
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

cwd = Path(__file__).parent

# LOCAL DATA:
model = ModelAPI(url="http://0.0.0.0:8000")  # without API
prediction = model.predict(cwd / "coco_sample.jpg")[0]  # Image path

# USING REMOTE DATA VIA API:
# model = ModelAPI(api=api, url="http://0.0.0.0:8000")
prediction = model.predict(image_id=4808207)[0]  # Image ID

prediction.visualize(cwd / "prediction_result.jpg")


################################
# 1. Serve with docker-compose #
################################
# Run the following command in the terminal:
# docker-compose up


################################
# 2. Run with docker run       #
################################
# Run the following command in the terminal:
# Pretrained
# docker run \
#   --shm-size=1g \
#   --runtime=nvidia \
#   --env PYTHONPATH=/app \
#   -p 8000:8000 \
#   supervisely/yolo:1.0.24-deploy \
#   deploy
#   --model "YOLO11n-det"

# Custom
# docker run \
#   --shm-size=1g \
#   --runtime=nvidia \
#   --env-file ~/supervisely.env \
#   --env PYTHONPATH=/app \
#   -v "./47653_YOLO:/model" \
#   -p 8000:8000 \
#   supervisely/yolo:1.0.24-deploy \
#   deploy \
#   --model "/model/checkpoints/best.pt" \
#   --device "cuda:0"


################################
# 3. Run locally               #
################################
# Run the following command in the terminal:
# PYTHONPATH="${PWD}:${PYTHONPATH}" \
# python ./supervisely_integration/serve/main.py deploy \
