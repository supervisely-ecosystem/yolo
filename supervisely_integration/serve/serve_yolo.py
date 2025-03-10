import os
from threading import Event
from typing import Any, Dict, Generator, List, Union

import cv2
import numpy as np
from ultralytics import YOLO

import supervisely as sly
from supervisely.convert.image.yolo.yolo_helper import SLY_YOLO_TASK_TYPE_MAP
from supervisely.nn.inference import ModelPrecision, ModelSource, RuntimeType, TaskType
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask

SERVE_PATH = "supervisely_integration/serve"


class YOLOModel(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "YOLO"
    MODELS = "supervisely_integration/models.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.yaml"

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        self.model_precision = ModelPrecision.FP32
        checkpoint_path = model_files["checkpoint"]
        if self.model_source == ModelSource.PRETRAINED:
            model_meta = model_info.get("meta", {})
            self.task_type = model_meta.get("task_type")
        else:
            self.task_type = model_info.get("task_type")

        if runtime == RuntimeType.PYTORCH:
            self.model = self._load_pytorch(checkpoint_path)
        elif runtime == RuntimeType.ONNXRUNTIME:
            self._check_onnx_device(device)
            self.model = self._load_onnx(checkpoint_path)
        elif runtime == RuntimeType.TENSORRT:
            self._check_tensorrt_device(device)
            self.model = self._load_tensorrt(checkpoint_path)
            self.max_batch_size = 1

        self.classes = list(self.model.names.values())
        self._load_model_meta()

    def _create_label(self, dto: Union[PredictionMask, PredictionBBox]):
        if self.task_type == TaskType.OBJECT_DETECTION or dto.class_name.endswith("_bbox"):
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            geometry = sly.Rectangle(*dto.bbox_tlbr)
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        elif self.task_type == TaskType.INSTANCE_SEGMENTATION and not dto.class_name.endswith(
            "_bbox"
        ):
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            if isinstance(dto, PredictionMask):
                if not dto.mask.any():  # skip empty masks
                    sly.logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
                    return None
                geometry = sly.Bitmap(dto.mask, extra_validation=False)
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        return label

    def _to_dto(self, prediction, settings: dict) -> List[Union[PredictionMask, PredictionBBox]]:
        """Converts YOLO Results to a List of Prediction DTOs."""
        dtos = []
        if self.task_type == TaskType.OBJECT_DETECTION:
            boxes_data = prediction.boxes.data
            for box in boxes_data:
                left, top, right, bottom, confidence, cls_index = (
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                    float(box[4]),
                    int(box[5]),
                )
                bbox = [top, left, bottom, right]
                dtos.append(PredictionBBox(self.classes[cls_index], bbox, confidence))
        elif self.task_type == TaskType.INSTANCE_SEGMENTATION:
            boxes_data = prediction.boxes.data
            if prediction.masks:
                masks = prediction.masks.data
                for box, mask in zip(boxes_data, masks):
                    left, top, right, bottom, confidence, cls_index = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    mask = mask.cpu().numpy()
                    mask_class_name = self.classes[cls_index]
                    dtos.append(PredictionMask(mask_class_name, mask, confidence))
                    bbox_class_name = self.classes[cls_index] + "_bbox"
                    bbox = [top, left, bottom, right]
                    dtos.append(PredictionBBox(bbox_class_name, bbox, confidence))
        return dtos

    def predict_video(self, video_path: str, settings: Dict[str, Any], stop: Event) -> Generator:
        retina_masks = self.task_type == TaskType.INSTANCE_SEGMENTATION
        predictions_generator = self.model(
            source=video_path,
            stream=True,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        for prediction in predictions_generator:
            if stop.is_set():
                predictions_generator.close()
                return
            yield self._to_dto(prediction, settings)

    def predict_benchmark(self, images_np: List[np.ndarray], settings: Dict):
        # RGB to BGR
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
        retina_masks = self.task_type == TaskType.INSTANCE_SEGMENTATION
        predictions = self.model(
            source=images_np,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        n = len(predictions)
        first_benchmark = predictions[0].speed
        # YOLO returns avg time per image, so we need to multiply it by the number of images
        benchmark = {
            "preprocess": first_benchmark["preprocess"] * n,
            "inference": first_benchmark["inference"] * n,
            "postprocess": first_benchmark["postprocess"] * n,
        }
        with sly.nn.inference.Timer() as timer:
            predictions = [self._to_dto(prediction, settings) for prediction in predictions]
        to_dto_time = timer.get_time()
        benchmark["postprocess"] += to_dto_time
        return predictions, benchmark

    def _check_onnx_device(self, device: str):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" not in providers:
            raise ValueError(
                f"Selected {device} device, but CUDAExecutionProvider is not available"
            )
        elif device == "cpu" and "CPUExecutionProvider" not in providers:
            raise ValueError(f"Selected {device} device, but CPUExecutionProvider is not available")

    def _check_tensorrt_device(self, device: str):
        if "cuda" not in device:
            raise ValueError(f"Selected '{device}' device, but TensorRT only supports CUDA devices")

    def _load_pytorch(self, weights_path: str):
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    def _load_onnx(self, weights_path: str):
        return self._load_runtime(weights_path, "onnx", dynamic=True)

    def _load_tensorrt(self, weights_path: str):
        return self._load_runtime(weights_path, "engine", dynamic=False)

    def _load_runtime(self, weights_path: str, format: str, **kwargs):
        def export_model():
            sly.logger.info(f"Exporting model to '{format}' format...")
            if self.gui is not None:
                bar = self.gui.download_progress(
                    message=f"Exporting model to '{format}' format...", total=1
                )
                self.gui.download_progress.show()
            model = YOLO(weights_path)
            model.export(format=format, **kwargs)
            if self.gui is not None:
                bar.update(1)
                self.gui.download_progress.hide()

        if weights_path.endswith(".pt"):
            exported_weights_path = weights_path.replace(".pt", f".{format}")
            file_name = sly.fs.get_file_name(weights_path)
            if file_name == "best" or not os.path.exists(exported_weights_path):
                export_model()
        else:
            exported_weights_path = weights_path

        model = YOLO(exported_weights_path, task=SLY_YOLO_TASK_TYPE_MAP[self.task_type])
        model.to(self.device)
        return model

    def _load_model_meta(self):
        self.class_names = list(self.model.names.values())
        if self.task_type == TaskType.OBJECT_DETECTION:
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
        elif self.task_type == TaskType.INSTANCE_SEGMENTATION:
            self.general_class_names = list(self.model.names.values())
            bbox_class_names = [f"{cls}_bbox" for cls in self.class_names]
            mask_class_names = self.class_names
            self.class_names = bbox_class_names + mask_class_names
            bbox_obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in bbox_class_names]
            mask_obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in mask_class_names]
            obj_classes = bbox_obj_classes + mask_obj_classes
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()

    def get_info(self):
        info = super().get_info()
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        return info
