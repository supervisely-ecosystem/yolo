import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely_integration.serve.serve_yolo import YOLOModel

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

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
#   supervisely/yolo:1.0.9-deploy \
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
#   supervisely/yolo:1.0.9-deploy \
#   deploy \
#   --model "/model/checkpoints/best.pt" \
#   --device "cuda:0"



################################
# 3. Run locally               #
################################
# Run the following command in the terminal:
# PYTHONPATH="${PWD}:${PYTHONPATH}" \
# python ./supervisely_integration/serve/main.py deploy \

###################################
# How to use the inference session: #
###################################

# Connect to the inference session
session = sly.nn.inference.Session(api, session_url="http://0.0.0.0:8000")

img_path = "supervisely_integration/demo/img/coco_sample.jpg"
ann = session.inference_image_path(img_path)

# Display the annotated image
img_preview_path = "supervisely_integration/demo/img/coco_sample_ann_preview.jpg"

if os.path.exists(img_preview_path):
    os.remove(img_preview_path)

# Read the image and draw the annotation
img = sly.image.read(img_path)
ann.draw_pretty(img)
sly.image.write(img_preview_path, img)
