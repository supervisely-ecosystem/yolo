import os

from dotenv import load_dotenv

from supervisely.nn import ModelSource, RuntimeType
from supervisely_integration.serve.serve_yolo import YOLOModel

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


model = YOLOModel()
model.serve()

model_info = model.pretrained_models[0]

model._load_model_headless(
    model_files={
        "checkpoint": os.path.expanduser("~/.cache/supervisely/checkpoints/yolov11n.pt"),
    },
    model_info=model_info,
    model_source=ModelSource.PRETRAINED,
    device="cuda",
    runtime=RuntimeType.PYTORCH,
)
