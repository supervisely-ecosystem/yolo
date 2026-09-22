"""Convert a pose project to the YOLO format against one project-wide keypoint template.

``Project.to_yolo`` writes each keypoint row with one slot per node of the label's own
class, while ``kpt_shape`` in ``data_config.yaml`` is taken from the largest keypoint
class of the project. A project with several keypoint classes of different sizes
therefore produces rows of different lengths, ultralytics rejects every image that
holds a short row ("labels require N columns each", or an inhomogeneous shape error
for an image that mixes classes), and those annotations never reach training. Node
order is the second casualty: a figure missing some of its template's nodes has the
points it does have written at the wrong keypoint indices.

Here every row is written against one template shared by the whole project: the node
labels of all keypoint classes, deduplicated and kept in the order they were first
seen. Its length is ``kpt_shape``, the index of a node label in it is the slot that
node is written at, and a node the figure does not have is written as ``0 0 0``.

Everything else -- the directory layout, the train/val split, the image naming and the
rest of ``data_config.yaml`` -- is the SDK's, so the converted dataset is the one the
app has always produced apart from the keypoint columns.
"""

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import yaml
from tqdm import tqdm

from supervisely._utils import generate_free_name
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.obj_class import ObjClass
# The three underscored names carry the semantics of the disabled_keypoints hyperparameter.
# They are imported rather than copied so that pose cannot drift from what that setting
# documents and from what detect and segment do; the SDK version is pinned by docker_image.
from supervisely.convert.image.yolo.yolo_helper import (
    DisabledKeypointsMode,
    YOLOTaskType,
    _disabled_keypoint_visibility,
    _disabled_keypoints_mode_for_class,
    _validate_disabled_keypoints,
    save_yolo_config,
)
from supervisely.geometry.graph import GraphNodes, Node
from supervisely.io.fs import get_file_name, get_file_name_with_ext, touch
from supervisely.project.project import Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly


def node_labels(obj_class: ObjClass) -> Dict[str, str]:
    """Node id to node label, in template order, for one keypoint class.

    A node with no label of its own is identified by its node id, which keeps it
    distinct from every node of every other class.
    """
    nodes_field = obj_class.geometry_type.items_json_field
    nodes = (obj_class.geometry_config or {}).get(nodes_field, {})
    return {node_id: node.get("label") or node_id for node_id, node in nodes.items()}


def get_keypoint_labels(meta: ProjectMeta) -> List[str]:
    """The keypoint template the whole project is converted against.

    Node labels of every keypoint class, deduplicated by label and kept in the order
    they were first seen. Its length is the ``kpt_shape`` of the dataset and the index
    of a label in it is the keypoint index that node is written at.
    """
    kpt_labels = []
    for obj_class in meta.obj_classes:
        if not issubclass(obj_class.geometry_type, GraphNodes):
            continue
        for node_label in node_labels(obj_class).values():
            if node_label not in kpt_labels:
                kpt_labels.append(node_label)
    return kpt_labels


def keypoints_to_yolo_line(
    class_idx: int,
    geometry: GraphNodes,
    img_height: int,
    img_width: int,
    kpt_labels: List[str],
    class_node_labels: Dict[str, str],
    disabled_keypoint_visibility: int = 1,
) -> str:
    """One YOLO pose line: the bounding box, then one slot per template keypoint.

    :param disabled_keypoint_visibility: Visibility flag written for a disabled node, as in
                                         the SDK converter. 1 keeps the node in the keypoint
                                         loss, 0 excludes it.
    """
    bbox = geometry.to_bbox()
    x, y, w, h = bbox.center.col, bbox.center.row, bbox.width, bbox.height
    x, y, w, h = x / img_width, y / img_height, w / img_width, h / img_height

    line = f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    nodes_by_label: Dict[str, Node] = {}
    for node_id, node in geometry.nodes.items():
        nodes_by_label.setdefault(class_node_labels.get(node_id, node_id), node)

    for kpt_label in kpt_labels:
        node = nodes_by_label.get(kpt_label)
        if node is None:
            line += " 0 0 0"
            continue
        visible = 2 if not node.disabled else disabled_keypoint_visibility
        line += (
            f" {node.location.col / img_width:.6f} {node.location.row / img_height:.6f} {visible}"
        )
    return line


def ann_to_yolo_lines(
    ann: Annotation,
    class_names: List[str],
    kpt_labels: List[str],
    node_labels_by_class: Dict[str, Dict[str, str]],
    disabled_keypoints: Union[str, Dict[str, str]] = DisabledKeypointsMode.INCLUDE,
) -> List[str]:
    """YOLO pose lines of one annotation. Labels that are not keypoints are skipped."""
    img_height, img_width = ann.img_size
    lines = []
    for label in ann.labels:
        class_name = label.obj_class.name
        class_node_labels = node_labels_by_class.get(class_name)
        if class_node_labels is None:
            continue
        mode = _disabled_keypoints_mode_for_class(disabled_keypoints, class_name)
        lines.append(
            keypoints_to_yolo_line(
                class_idx=class_names.index(class_name),
                geometry=label.geometry,
                img_height=img_height,
                img_width=img_width,
                kpt_labels=kpt_labels,
                class_node_labels=class_node_labels,
                disabled_keypoint_visibility=_disabled_keypoint_visibility(mode),
            )
        )
    return lines


def save_pose_yolo_config(meta: ProjectMeta, dest_dir: Path, num_keypoints: int) -> None:
    """Write ``data_config.yaml``, with ``kpt_shape`` taken from the shared template."""
    save_yolo_config(meta, str(dest_dir), with_keypoint=True)
    config_path = dest_dir / "data_config.yaml"
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)
    data_config["kpt_shape"] = [num_keypoints, 3]
    data_config["flip_idx"] = list(range(num_keypoints))
    with open(config_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=None)


def convert(
    project: Project,
    dest_dir: str,
    val_datasets: Optional[List[str]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    disabled_keypoints: Union[str, Dict[str, str]] = DisabledKeypointsMode.INCLUDE,
) -> str:
    """Convert a Supervisely project to a YOLO pose dataset in ``dest_dir``.

    :param disabled_keypoints: How disabled graph nodes are exported, exactly as in
                               ``Project.to_yolo``. Either a :class:`DisabledKeypointsMode`
                               value applied to every class, or a dict mapping a class name
                               to one. Classes missing from the dict use "include".
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    meta = project.meta
    _validate_disabled_keypoints(disabled_keypoints, [c.name for c in meta.obj_classes])
    kpt_labels = get_keypoint_labels(meta)
    if len(kpt_labels) == 0:
        # nothing to share a template between, the SDK converter is already correct
        project.to_yolo(
            dest_dir,
            YOLOTaskType.POSE,
            log_progress=log_progress,
            progress_cb=progress_cb,
            val_datasets=val_datasets,
            disabled_keypoints=disabled_keypoints,
        )
        return dest_dir

    logger.info(
        f"Keypoint template of the dataset: {kpt_labels}",
        extra={"kpt_shape": [len(kpt_labels), 3]},
    )
    save_pose_yolo_config(meta, dest_path, len(kpt_labels))

    split_dirs = {}
    for split in ("train", "val"):
        images_dir = dest_path / "images" / split
        labels_dir = dest_path / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = (images_dir, labels_dir)

    if progress_cb is not None:
        log_progress = False
    if log_progress:
        progress_cb = tqdm_sly(
            desc="Converting Supervisely project to YOLO format", total=project.total_items
        ).update

    class_names = [obj_class.name for obj_class in meta.obj_classes]
    node_labels_by_class = {
        obj_class.name: node_labels(obj_class)
        for obj_class in meta.obj_classes
        if issubclass(obj_class.geometry_type, GraphNodes)
    }

    used_names = set()
    for dataset in project.datasets:
        is_val = None if val_datasets is None else dataset.name in val_datasets
        for name in dataset.get_items_names():
            ann = Annotation.load_json_file(dataset.get_ann_path(name), meta)
            if is_val is None:
                split = "val" if ann.img_tags.get("val") else "train"
            else:
                split = "val" if is_val else "train"
            images_dir, labels_dir = split_dirs[split]

            img_path = dataset.get_img_path(name)
            img_name = f"{dataset.short_name}_{get_file_name_with_ext(img_path)}"
            img_name = generate_free_name(
                used_names, img_name, with_ext=True, extend_used_names=True
            )
            shutil.copy2(img_path, images_dir / img_name)

            label_path = str(labels_dir / f"{get_file_name(img_name)}.txt")
            yolo_lines = ann_to_yolo_lines(
                ann, class_names, kpt_labels, node_labels_by_class, disabled_keypoints
            )
            if len(yolo_lines) > 0:
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
            else:
                touch(label_path)

            if progress_cb is not None:
                progress_cb(1)
        logger.info(f"Dataset '{dataset.short_name}' has been converted to YOLO format.")
    logger.info(f"Project '{project.name}' has been converted to YOLO format.")

    return dest_dir
