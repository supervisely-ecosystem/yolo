import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
from supervisely.nn.prediction_dto import PredictionBBox

SERVE_PATH = "supervisely_integration/serve"


class YOLOServe(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "YOLO"
    MODELS = "supervisely_integration/models.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.yaml"

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        pass

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        pass

    @torch.no_grad()
    def _predict_pytorch(
        self, images_np: List[np.ndarray], settings: dict = None
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        pass

    def _predict_onnx(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        pass

    @torch.no_grad()
    def _predict_tensorrt(self, images_np: List[np.ndarray], settings: dict):
        pass

    def _prepare_input(self, images_np: List[np.ndarray], device=None):
        pass

    def _format_prediction(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_tresh: float
    ) -> List[PredictionBBox]:
        pass

    def _format_predictions(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, settings: dict
    ) -> List[List[PredictionBBox]]:
        pass
