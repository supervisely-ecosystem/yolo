import os

import numpy as np
from PIL import Image

from supervisely.nn import ModelSource, RuntimeType
from supervisely_integration.serve.serve_yolo import YOLOModel

model = YOLOModel()

model_info = model.pretrained_models[0]

model._load_model_headless(
    model_files={
        "checkpoint": os.path.expanduser("~/.cache/supervisely/checkpoints/yolov11n.pt"),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.TENSORRT,
)

image = Image.open("supervisely_integration/serve/scripts/coco_sample.jpg").convert("RGB")
img = np.array(image)

ann = model._inference_auto([img], {"confidence_threshold": 0.5})[0][0]

ann.draw_pretty(img)
Image.fromarray(img).save("supervisely_integration/serve/scripts/predict.jpg")
