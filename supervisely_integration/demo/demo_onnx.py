from os.path import join

import torch
from ultralytics import YOLO

from supervisely.io.fs import get_file_ext, get_file_name

# Predict settings
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "detect"

# Put your files here
demo_dir = join("supervisely_integration", "demo")
checkpoint_name = "best.onnx"
checkpoint_path = join(demo_dir, "model", checkpoint_name)
image_path = join(demo_dir, "img", "coco_sample.jpg")
result_path = join(demo_dir, "img", f"result_{get_file_name(image_path)}{get_file_ext(image_path)}")

# Load model and predict
model = YOLO(checkpoint_path, task)
results = model.predict(source=image_path, device=device)
for result in results:
    result.save(filename=result_path)
