import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.nn.utils import ModelSource, RuntimeType

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 68910  # <---- Change this to your task_id
method = "deploy_from_api"


# Pretrained
pretrained_model_data = {
    "deploy_params": {
        "model_files": {
            "checkpoint": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        },
        "model_source": ModelSource.PRETRAINED,
        "model_info": {
            "Model": "YOLO11n-det",
            "Size (pixels)": "640",
            "mAP": "39.5",
            "params (M)": "2.6",
            "FLOPs (B)": "6.5",
            "meta": {
                "task_type": "object detection",
                "model_name": "yolo11n",
                "model_files": {
                    "checkpoint": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
                },
            },
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

# Custom
custom_model_data = {
    "deploy_params": {
        "model_files": {
            "checkpoint": "/experiments/43192_Apples/71505_YOLO/checkpoints/best.pth",
        },
        "model_source": ModelSource.CUSTOM,
        "model_info": {
            "artifacts_dir": "/experiments/43192_Apples/71505_YOLO",
            "framework_name": "YOLO",
            "model_name": "yolov11n",
            "model_meta": "model_meta.json",
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

api.app.send_request(task_id, method, custom_model_data)
