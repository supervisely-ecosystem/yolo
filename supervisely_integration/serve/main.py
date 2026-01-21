import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely_integration.serve.serve_yolo import YOLOModel

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

model = YOLOModel(
    use_gui=True,
    use_serving_gui_template=True,
)
model.gui.pretrained_models_table.set_active_row(1)
model.serve()
