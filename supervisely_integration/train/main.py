from os import getcwd, rename
from os.path import join

from ultralytics import YOLO
from ultralytics.utils import SettingsManager

from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.nn import ModelSource, TaskType
from supervisely.nn.training.train_app import TrainApp
from supervisely_integration.serve.serve_yolo import YOLOModel
from supervisely_integration.train.trainer import Trainer  # Import the Trainer class

# TODOs
# - Add support for segmentation and pose estimation
# - [Low priority] Export ONNX runtime library
# - [Low priority] Check YOLO conversion for each task type

sly_yolo_task_map = {
    TaskType.OBJECT_DETECTION: "detect",
    TaskType.INSTANCE_SEGMENTATION: "segment",
    TaskType.POSE_ESTIMATION: "pose",
}

base_path = "supervisely_integration/train"
train = TrainApp(
    "YOLO",
    f"supervisely_integration/models.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)

train.register_inference_class(YOLOModel)
train.gui.load_from_app_state("supervisely_integration/train/app_state.json")


@train.start
def start_training():
    """Start the training process with a clean, high-level workflow."""
    data_config_path = convert_data()
    train_config = prepare_train_config(data_config_path)

    log_dir = join(getcwd(), train_config["project"], train_config["name"])
    train.start_tensorboard(log_dir)
    trainer = Trainer(train_config)
    trainer.train()

    output_checkpoint_dir = join(getcwd(), train_config["project"], train_config["name"], "weights")
    experiment_info = {
        "model_name": train.model_name,
        "model_files": {},
        "checkpoints": output_checkpoint_dir,
        "best_checkpoint": "best.pt",
    }
    return experiment_info


@train.export_onnx
def to_onnx(experiment_info: dict):
    """Export the model to ONNX format."""
    return export_checkpoint(
        experiment_info["best_checkpoint"], format="onnx", fp16=False, dynamic=False
    )


@train.export_tensorrt
def to_tensorrt(experiment_info: dict):
    """Export the model to TensorRT format."""
    return export_checkpoint(
        experiment_info["best_checkpoint"], format="engine", fp16=False, dynamic=False
    )


def convert_data():
    """Convert Supervisely project data to YOLO format."""
    project = train.sly_project
    yolo_project_path = join(getcwd(), train.work_dir, "yolo_project")
    project.to_yolo(yolo_project_path, train.task_type, val_datasets=["val"])
    data_config_path = join(yolo_project_path, "data_config.yaml")

    # Update YOLO settings
    weights_dir = join(getcwd(), train.model_dir)
    runs_dir = join(getcwd(), train.output_dir, "runs")
    datasets_dir = yolo_project_path
    yolo_settings = SettingsManager("supervisely_integration/train/yolo_settings.json")
    yolo_settings.update(weights_dir=weights_dir, runs_dir=runs_dir, datasets_dir=datasets_dir)
    return data_config_path


def prepare_train_config(data_config_path):
    """Prepare the training configuration dictionary."""
    if train.model_source == ModelSource.PRETRAINED:
        checkpoint_path = join(
            getcwd(), train.model_dir, get_file_name(train.model_files["checkpoint"])
        )
    else:
        checkpoint_path = join(
            getcwd(), train.model_dir, get_file_name_with_ext(train.model_files["checkpoint"])
        )

    train_config = {**train.hyperparameters}
    train_config.update(
        {
            "task": sly_yolo_task_map[train.task_type],
            "mode": "train",
            "model": checkpoint_path,
            "data": data_config_path,
            "device": train.device,
            "project": join(getcwd(), train.output_dir),
            "name": "ultralytics",
            "cache": False,
        }
    )
    return train_config


def export_checkpoint(checkpoint_path: str, format: str, fp16=False, dynamic=False):
    """Export a checkpoint to the specified format."""
    exported_checkpoint_path = checkpoint_path.replace(".pt", f".{format}")
    if fp16:
        exported_checkpoint_path = exported_checkpoint_path.replace(f".{format}", f"_fp16.{format}")
    model = YOLO(checkpoint_path)
    model.export(format=format, half=fp16, dynamic=dynamic)
    if fp16:
        rename(checkpoint_path.replace(".pt", f".{format}"), exported_checkpoint_path)
        if format == "engine":
            rename(
                checkpoint_path.replace(".pt", f".onnx"),
                exported_checkpoint_path.replace(".engine", ".onnx"),
            )
    return exported_checkpoint_path
