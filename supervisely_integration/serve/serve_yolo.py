import os
from threading import Event
from typing import Any, Dict, Generator, List, Union

import cv2
import numpy as np
from ultralytics import YOLO

import supervisely as sly
from supervisely.convert.image.yolo.yolo_helper import (
    SLY_YOLO_TASK_TYPE_MAP,
    create_geometry_config,
)
from supervisely.nn.inference import ModelPrecision, ModelSource, RuntimeType, TaskType
from supervisely.nn.prediction_dto import (
    PredictionBBox,
    PredictionKeypoints,
    PredictionMask,
)
from supervisely_integration.serve.keypoints_confidence import (
    count_visible,
    select_visible_indices,
)
from supervisely_integration.serve.keypoints_template import human_template

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
            self.model = self._load_onnx(checkpoint_path, device)
        elif runtime == RuntimeType.TENSORRT:
            self.model = self._load_tensorrt(checkpoint_path, device)
            self.max_batch_size = 1

        expected_yolo_task = SLY_YOLO_TASK_TYPE_MAP.get(self.task_type)
        if expected_yolo_task is not None and self.model.task != expected_yolo_task:
            raise ValueError(
                f"Checkpoint is a '{self.model.task}' model, but the selected task type is "
                f"'{self.task_type}' (YOLO task '{expected_yolo_task}'). "
                f"Select a checkpoint that matches the task type."
            )

        self.classes = list(self.model.names.values())
        self._load_model_meta()

    def get_info(self):
        info = super().get_info()
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        return info

    # Loaders --------------- #
    def _load_pytorch(self, checkpoint_path: str):
        model = YOLO(checkpoint_path)
        model.to(self.device)
        return model

    def _load_onnx(self, checkpoint_path: str, device: str):
        self._check_onnx_device(device)
        model = YOLO(checkpoint_path, task=SLY_YOLO_TASK_TYPE_MAP[self.task_type])
        return model

    def _load_tensorrt(self, checkpoint_path: str, device: str):
        self._check_tensorrt_device(device)
        model = YOLO(checkpoint_path, task=SLY_YOLO_TASK_TYPE_MAP[self.task_type])
        return model

    # -------------------------- #

    # Predictions ----------- #
    def predict_video(self, video_path: str, settings: Dict[str, Any], stop: Event) -> Generator:
        retina_masks = self.task_type == TaskType.INSTANCE_SEGMENTATION
        predictions_generator = self.model(
            source=video_path,
            stream=True,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.model.device,
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
            device=self.model.device,
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

    def _create_label(self, dto: Union[PredictionMask, PredictionBBox, PredictionKeypoints]):
        if self.task_type == TaskType.POSE_ESTIMATION:
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            disabled_flags = getattr(dto, "disabled", None) or []
            nodes = []
            for i, (node_key, (x, y)) in enumerate(zip(dto.labels, dto.coordinates)):
                is_disabled = bool(disabled_flags[i]) if i < len(disabled_flags) else False
                nodes.append(sly.Node(label=node_key, row=y, col=x, disabled=is_disabled))
            geometry = sly.GraphNodes(nodes)
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        elif self.task_type == TaskType.OBJECT_DETECTION or dto.class_name.endswith("_bbox"):
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

    def _to_dto(
        self, prediction, settings: dict
    ) -> List[Union[PredictionMask, PredictionBBox, PredictionKeypoints]]:
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
                    confidence = float(box[4])
                    cls_index = int(box[5])
                    mask = mask.cpu().numpy()
                    class_name = self.classes[cls_index]
                    dtos.append(PredictionMask(class_name, mask, confidence))
        elif self.task_type == TaskType.POSE_ESTIMATION:
            boxes_data = prediction.boxes.data
            if prediction.keypoints is not None:
                point_threshold = settings.get("point_threshold", 0.1)
                # when set, every point of the class template is emitted; the ones
                # scoring below `point_threshold` are kept but marked disabled instead
                # of being dropped from the graph, which is what a manually annotated
                # not-visible keypoint looks like. A node that is absent carries no
                # point label and no skeleton edge can reach it.
                keep_all_keypoints = settings.get("keep_all_keypoints", False)
                image_height, image_width = prediction.orig_shape
                keypoints_data = prediction.keypoints.data
                # (x, y, visibility) per point, or (x, y) when the checkpoint
                # carries no per-point confidence
                with_scores = keypoints_data.shape[-1] == 3
                for box, keypoints in zip(boxes_data, keypoints_data):
                    confidence = float(box[4])
                    cls_index = int(box[5])
                    class_name = self.classes[cls_index]
                    node_keys = self.keypoint_node_keys[class_name]
                    if with_scores:
                        scores = [float(point[2]) for point in keypoints[: len(node_keys)]]
                        chosen, disabled = select_visible_indices(
                            scores, point_threshold, keep_all_keypoints
                        )
                    else:
                        # no per-point confidence to threshold on: keep every point
                        chosen = list(range(min(len(node_keys), len(keypoints))))
                        disabled = [False] * len(chosen)
                    if count_visible(disabled) < 1:  # a graph needs a visible point
                        continue
                    # coordinates are only moved off the GPU for points actually kept
                    labels = [node_keys[i] for i in chosen]
                    coordinates = [keypoints[i][:2].cpu().numpy() for i in chosen]
                    for position, is_disabled in enumerate(disabled):
                        if is_disabled:
                            # a low-confidence point often lands outside the image, and
                            # a graph with any node out of bounds is dropped whole when
                            # the annotation is built. A disabled node is not drawn, so
                            # pinning it to the image is safe and keeps the figure.
                            coordinates[position] = np.clip(
                                coordinates[position],
                                (0, 0),
                                (image_width - 1, image_height - 1),
                            )
                    dto = PredictionKeypoints(class_name, labels, coordinates)
                    dto.score = confidence
                    dto.disabled = disabled
                    dtos.append(dto)
        return dtos

    # -------------------------- #

    # Converters --------------- #
    def export_onnx(self, deploy_params: dict) -> dict:
        # @TODO: check how checkpoint_path is changed
        checkpoint_path = deploy_params["model_files"]["checkpoint"]
        model = YOLO(checkpoint_path)
        model.export(format="onnx", device=self.device, dynamic=True)
        return checkpoint_path

    def export_tensorrt(self, deploy_params: dict) -> dict:
        # @TODO: check how checkpoint_path is changed
        checkpoint_path = deploy_params["model_files"]["checkpoint"]
        model = YOLO(checkpoint_path)
        model.export(format="engine", device=self.device, dynamic=False)
        return checkpoint_path

    # -------------------------- #

    # Utils -------------------- #
    def _load_model_meta(self):
        self.class_names = list(self.model.names.values())
        if self.task_type == TaskType.OBJECT_DETECTION:
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
        elif self.task_type == TaskType.INSTANCE_SEGMENTATION:
            self.general_class_names = list(self.model.names.values())
            obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.class_names]
        elif self.task_type == TaskType.POSE_ESTIMATION:
            obj_classes = self._pose_obj_classes()
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()
        if self.task_type == TaskType.POSE_ESTIMATION:
            # node keys of each class template, in the order the model predicts them.
            # A template built here keys its nodes by point label, one that came from a
            # project meta keys them by node id -- either way the key is what a Node has
            # to carry for the graph to validate against the class.
            self.keypoint_node_keys = {
                obj_class.name: list(obj_class.geometry_config[sly.GraphNodes.items_json_field])
                for obj_class in obj_classes
            }
            self._check_keypoint_count()

    def _pose_obj_classes(self) -> List[sly.ObjClass]:
        """Build a keypoints class per model class, each with its graph template.

        A custom checkpoint carries the template of the project it was trained on in
        its model meta, which the SDK has already loaded by this point. A pretrained
        checkpoint is always COCO-pose, so it gets the human template; anything else
        falls back to an unnamed template of the right size.
        """
        trained_meta = self._model_meta
        obj_classes = []
        for name in self.class_names:
            geometry_config = None
            if trained_meta is not None:
                trained_class = trained_meta.get_obj_class(name)
                if trained_class is not None and trained_class.geometry_type == sly.GraphNodes:
                    geometry_config = trained_class.geometry_config
            if geometry_config is None:
                if self.model_source == ModelSource.PRETRAINED:
                    geometry_config = human_template
                else:
                    num_keypoints = self._num_keypoints()
                    if num_keypoints is None:
                        raise ValueError(
                            f"Checkpoint has no keypoints template for class '{name}' and the "
                            f"number of keypoints cannot be read from it. Deploy a checkpoint "
                            f"trained in Supervisely, which carries the template in its model meta."
                        )
                    geometry_config = create_geometry_config(num_keypoints)
            obj_classes.append(
                sly.ObjClass(name, sly.GraphNodes, geometry_config=geometry_config)
            )
        return obj_classes

    def _num_keypoints(self) -> Union[int, None]:
        """Keypoints per object the loaded checkpoint predicts, or None if unreadable."""
        kpt_shape = getattr(getattr(self.model, "model", None), "kpt_shape", None)
        if kpt_shape is None:
            kpt_shape = (self.model.overrides or {}).get("kpt_shape")
        if kpt_shape is None:
            return None
        return int(kpt_shape[0])

    def _check_keypoint_count(self):
        """Warn when a class template does not describe every point the model predicts.

        The prediction is zipped against the template, so a shorter template silently
        drops trailing points instead of producing an obviously wrong graph.
        """
        num_keypoints = self._num_keypoints()
        if num_keypoints is None:
            return
        for class_name, node_keys in self.keypoint_node_keys.items():
            if len(node_keys) != num_keypoints:
                sly.logger.warning(
                    f"Class '{class_name}' has a {len(node_keys)}-point keypoints template, "
                    f"but the checkpoint predicts {num_keypoints} points per object."
                )

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

    # -------------------------- #
