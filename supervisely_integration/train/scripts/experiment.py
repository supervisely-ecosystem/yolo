import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.template.experiment.experiment_generator import ExperimentGenerator
from supervisely_integration.serve.main import YOLOModel

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

experiment_info = {
    "experiment_name": "15605_Animals (Rectangle)_YOLO12n",
    "framework_name": "YOLO",
    "model_name": "YOLO12n",
    "task_type": "object detection",
    "project_id": 1322,
    "task_id": 15605,
    "model_files": {},
    "checkpoints": [
        "checkpoints/best.pt",
        "checkpoints/epoch0.pt",
        "checkpoints/epoch10.pt",
        "checkpoints/last.pt",
    ],
    "best_checkpoint": "best.pt",
    "export": {"ONNXRuntime": "export/best.onnx", "TensorRT": "export/best.engine"},
    "app_state": "app_state.json",
    "model_meta": "model_meta.json",
    "hyperparameters": "hyperparameters.yaml",
    "artifacts_dir": "/experiments/1322_Animals (Rectangle)/15605_YOLO/",
    "datetime": "2025-06-02 13:49:41",
    "evaluation_report_id": None,
    "evaluation_report_link": None,
    "evaluation_metrics": {},
    "logs": {
        "type": "tensorboard",
        "link": "/experiments/1322_Animals (Rectangle)/15605_YOLO/logs/",
    },
    "train_val_split": "train_val_split.json",
    "train_size": 27,
    "val_size": 27,
}

model_meta = {
    "classes": [
        {
            "title": "cat",
            "description": "",
            "shape": "rectangle",
            "color": "#A80B10",
            "geometry_config": {},
            "id": 32406,
            "hotkey": "",
        },
        {
            "title": "dog",
            "description": "",
            "shape": "rectangle",
            "color": "#B8E986",
            "geometry_config": {},
            "id": 32407,
            "hotkey": "",
        },
        {
            "title": "horse",
            "description": "",
            "shape": "rectangle",
            "color": "#9F21DE",
            "geometry_config": {},
            "id": 32408,
            "hotkey": "",
        },
        {
            "title": "sheep",
            "description": "",
            "shape": "rectangle",
            "color": "#1EA49B",
            "geometry_config": {},
            "id": 32409,
            "hotkey": "",
        },
        {
            "title": "squirrel",
            "description": "",
            "shape": "rectangle",
            "color": "#F8E71C",
            "geometry_config": {},
            "id": 32410,
            "hotkey": "",
        },
    ],
    "tags": [
        {
            "name": "animal age group",
            "value_type": "oneof_string",
            "color": "#F5A623",
            "values": ["juvenile", "adult", "senior"],
            "id": 4781,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "animal age group_1",
            "value_type": "any_string",
            "color": "#8A0F59",
            "id": 4782,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "animal count",
            "value_type": "any_number",
            "color": "#E3BE1C",
            "id": 4783,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "cat",
            "value_type": "none",
            "color": "#A80B10",
            "id": 4784,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "dog",
            "value_type": "none",
            "color": "#B8E986",
            "id": 4785,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "horse",
            "value_type": "none",
            "color": "#9F21DE",
            "id": 4786,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "imgtag",
            "value_type": "none",
            "color": "#FF03D6",
            "id": 4787,
            "hotkey": "",
            "applicable_type": "imagesOnly",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "sheep",
            "value_type": "none",
            "color": "#1EA49B",
            "id": 4788,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "squirrel",
            "value_type": "none",
            "color": "#F8E71C",
            "id": 4789,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
    ],
    "projectType": "images",
    "projectSettings": {
        "multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}
    },
}
model_meta = sly.ProjectMeta.from_json(model_meta)

hyperparameters_yaml = """
# Learn more about YOLO hyperparameters: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml

# Train settings
epochs: 20 # (int) number of epochs to train for
patience: 50 # (int) epochs to wait for no observable improvement for early stopping of training
batch: 16 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 640 # (int | list) input images size as int for train and val modes, or list[h,w] for predict and export modes
save_period: 10 # (int) Save checkpoint every x epochs (disabled if < 1)
workers: 8 # (int) number of worker threads for data loading (per RANK if DDP)
optimizer: auto # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
verbose: False # (bool) whether to print verbose output
seed: 0 # (int) random seed for reproducibility
cos_lr: False # (bool) use cosine learning rate scheduler
close_mosaic: 10 # (int) disable mosaic augmentation for final epochs (0 to disable)
amp: True # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
freeze: None # (int | list, optional) freeze first n layers, or freeze list of layer indices during training

# Hyperparameters
lr0: 0.01 # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf: 0.01 # (float) final learning rate (lr0 * lrf)
momentum: 0.937 # (float) SGD momentum/Adam beta1
weight_decay: 0.0005 # (float) optimizer weight decay 5e-4
warmup_epochs: 3.0 # (float) warmup epochs (fractions ok)
warmup_momentum: 0.8 # (float) warmup initial momentum
warmup_bias_lr: 0.1 # (float) warmup initial bias lr
box: 7.5 # (float) box loss gain
cls: 0.5 # (float) cls loss gain (scale with pixels)
dfl: 1.5 # (float) dfl loss gain
pose: 12.0 # (float) pose loss gain
kobj: 1.0 # (float) keypoint obj loss gain

# Augmentations
hsv_h: 0.015 # (float) image HSV-Hue augmentation (fraction)
hsv_s: 0.7 # (float) image HSV-Saturation augmentation (fraction)
hsv_v: 0.4 # (float) image HSV-Value augmentation (fraction)
degrees: 0.0 # (float) image rotation (+/- deg)
translate: 0.1 # (float) image translation (+/- fraction)
scale: 0.5 # (float) image scale (+/- gain)
shear: 0.0 # (float) image shear (+/- deg)
perspective: 0.0 # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0 # (float) image flip up-down (probability)
fliplr: 0.5 # (float) image flip left-right (probability)
bgr: 0.0 # (float) image channel BGR (probability)
mosaic: 0.0 # (float) image mosaic (probability)
mixup: 0.0 # (float) image mixup (probability)
copy_paste: 0.0 # (float) segment copy-paste (probability)
copy_paste_mode: "flip" # (str) the method to do copy_paste augmentation (flip, mixup)
"""
app_options = {
    "demo": {
        "path": "supervisely_integration/demo",
    },
}


experiment = ExperimentGenerator(
    api=api,
    experiment_info=experiment_info,
    hyperparameters=hyperparameters_yaml,
    model_meta=model_meta,
    serving_class=YOLOModel,
    team_id=team_id,
    output_dir="./experiment_report",
    app_options=app_options,
)

experiment.generate()
experiment.upload_to_artifacts()
