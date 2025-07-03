<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/yolo/releases/download/v1.0.3/poster_train_yolo.jpg"/>

# Train YOLO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolo/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolo)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/yolo/supervisely_integration/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/yolo/supervisely_integration/train.png)](https://supervisely.com)

</div>

# Overview

This app allows you to train models using checkpoints from YOLO architecture on a selected dataset. You can define model checkpoints, data split methods, training hyperparameters and many other features related to model training. The app supports models pretrained on COCO or Open Images V7 dataset and models trained on custom datasets. Supported task types include object detection and instance segmentation. Support for pose estimation will be added in the upcoming releases, meanwhile you can use [Train YOLOv8](https://ecosystem.supervisely.com/apps/yolov8/train) app for pose estimation.

**Object Detection models**

| Model                        | Size (px) | mAP  | Params (M) | FLOPs (B) | Checkpoint                                                                                 |
| ---------------------------- | --------- | ---- | ---------- | --------- | ------------------------------------------------------------------------------------------ |
| YOLO12n                      | 640       | 40.6 | 2.6        | 6.5       | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt)      |
| YOLO12s                      | 640       | 48.0 | 9.3        | 21.4      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt)      |
| YOLO12m                      | 640       | 52.5 | 20.2       | 67.5      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt)      |
| YOLO12l                      | 640       | 53.7 | 26.4       | 88.9      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt)      |
| YOLO12x                      | 640       | 55.2 | 59.1       | 199.0     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt)      |
| YOLO11n-det                  | 640       | 39.5 | 2.6        | 6.5       | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)      |
| YOLO11s-det                  | 640       | 47.0 | 9.4        | 21.5      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)      |
| YOLO11m-det                  | 640       | 51.5 | 20.1       | 68.0      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)      |
| YOLO11l-det                  | 640       | 53.4 | 25.3       | 86.9      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)      |
| YOLO11x-det                  | 640       | 54.7 | 56.9       | 194.9     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)      |
| YOLOv10n-det                 | 640       | 39.5 | 2.3        | 6.7       | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt)     |
| YOLOv10s-det                 | 640       | 46.8 | 7.2        | 21.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt)     |
| YOLOv10m-det                 | 640       | 51.3 | 15.4       | 59.1      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt)     |
| YOLOv10l-det                 | 640       | 53.4 | 24.4       | 120.3     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt)     |
| YOLOv10x-det                 | 640       | 54.4 | 29.5       | 160.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt)     |
| YOLOv9c-det                  | 640       | 53.0 | 25.5       | 102.8     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt)      |
| YOLOv9e-det                  | 640       | 55.6 | 58.1       | 192.5     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt)      |
| YOLOv8n-det (COCO)           | 640       | 37.3 | 3.2        | 8.7       | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)      |
| YOLOv8n-det (Open Images V7) | 640       | 18.4 | 3.5        | 10.5      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt) |
| YOLOv8s-det (COCO)           | 640       | 44.9 | 11.2       | 28.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)      |
| YOLOv8s-det (Open Images V7) | 640       | 27.7 | 11.4       | 29.7      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt) |
| YOLOv8m-det (COCO)           | 640       | 50.2 | 25.9       | 78.9      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)      |
| YOLOv8m-det (Open Images V7) | 640       | 33.6 | 26.2       | 80.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt) |
| YOLOv8l-det (COCO)           | 640       | 52.9 | 43.7       | 165.2     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)      |
| YOLOv8l-det (Open Images V7) | 640       | 34.9 | 44.1       | 167.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt) |
| YOLOv8x-det (COCO)           | 640       | 53.9 | 68.2       | 257.8     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)      |
| YOLOv8x-det (Open Images V7) | 640       | 36.3 | 68.7       | 260.6     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt) |
| YOLOv5nu                     | 640       | 34.3 | 2.6        | 7.7       | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt)     |
| YOLOv5su                     | 640       | 43.0 | 9.1        | 24.0      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt)     |
| YOLOv5mu                     | 640       | 59.0 | 25.1       | 64.2      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5mu.pt)     |
| YOLOv5lu                     | 640       | 52.2 | 43.2       | 135.0     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5lu.pt)     |
| YOLOv5xu                     | 640       | 53.2 | 97.2       | 246.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5xu.pt)     |
| YOLOv5n6u                    | 1280      | 42.1 | 4.3        | 7.8       | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5n6u.pt)    |
| YOLOv5s6u                    | 1280      | 48.6 | 15.3       | 24.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5s6u.pt)    |
| YOLOv5m6u                    | 1280      | 53.6 | 41.2       | 65.7      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5m6u.pt)    |
| YOLOv5l6u                    | 1280      | 55.7 | 86.1       | 137.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5l6u.pt)    |
| YOLOv5x6u                    | 1280      | 56.8 | 155.4      | 250.0     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5x6u.pt)    |

**Instance Segmentation models**

| Model       | Size (px) | mAP (box) | mAP (mask) | Params (M) | FLOPs (B) | Checkpoint                                                                                |
| ----------- | --------- | --------- | ---------- | ---------- | --------- | ----------------------------------------------------------------------------------------- |
| YOLO11n-seg | 640       | 38.9      | 32.0       | 2.9        | 10.4      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) |
| YOLO11s-seg | 640       | 46.6      | 37.8       | 10.1       | 35.5      | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) |
| YOLO11m-seg | 640       | 51.5      | 41.5       | 22.4       | 123.3     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) |
| YOLO11l-seg | 640       | 53.4      | 42.9       | 27.6       | 142.2     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) |
| YOLO11x-seg | 640       | 54.7      | 43.8       | 62.1       | 319.0     | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) |
| YOLOv9c-seg | 640       | 52.4      | 42.2       | 27.9       | 159.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt) |
| YOLOv9e-seg | 640       | 55.1      | 44.3       | 60.5       | 248.4     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt) |
| YOLOv8n-seg | 640       | 36.7      | 30.5       | 3.4        | 12.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt) |
| YOLOv8s-seg | 640       | 44.6      | 36.8       | 11.8       | 42.6      | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt) |
| YOLOv8m-seg | 640       | 49.9      | 40.8       | 27.3       | 110.2     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.pt) |
| YOLOv8l-seg | 640       | 52.3      | 42.6       | 46.0       | 220.5     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-seg.pt) |
| YOLOv8x-seg | 640       | 53.4      | 43.4       | 71.8       | 344.1     | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt) |

# How to Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/user-attachments/assets/2b5f867e-7d99-4c06-9a4c-96ba29523941" width="100%" style='padding-top: 10px'>

**Step 2.** Select train / val split

<img src="https://github.com/user-attachments/assets/cb50e8e1-9e10-4195-b000-c694358f5ab1" width="100%" style='padding-top: 10px'>

**Step 3.** Select the classes you want to train on

<img src="https://github.com/user-attachments/assets/7b5d22a8-6c43-4b3f-b262-01a9ed2e58d4" width="100%" style='padding-top: 10px'>

**Step 4.** Select the model you want to train

The screenshot doesn’t include all the available models. You can view the complete list in the [overview](#overview) section.

<img src="https://github.com/user-attachments/assets/fa4df097-4e8d-48ae-b53d-44ff8a7beb5a" width="100%" style='padding-top: 10px'>

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/user-attachments/assets/4cf73b4e-f3c8-47db-91c6-7145c2352274" width="100%" style='padding-top: 10px'>

**Step 6.** Enter experiment name and start training

<img src="https://github.com/user-attachments/assets/fdb2943d-7ee2-4ff8-8616-3587911ce270" width="100%" style='padding-top: 10px'>

**Step 7.** Monitor training progress

<img src="https://github.com/user-attachments/assets/41f2d8b3-dd5a-4f4c-9605-1a3fa5e89827" width="100%" style='padding-top: 10px'>

# Obtain saved checkpoints

All trained checkpoints that are generated through the training process are stored in [Team Files](https://app.supervisely.com/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/10a7a32a-8adc-4cc0-b05d-ff1ef8e08552" width="100%" style='padding-top: 10px'>

# How to use your checkpoints outside Supervisely Platform

After you've trained a model in Supervisely, you can download the checkpoint from Team Files and use it as a simple PyTorch model without Supervisely Platform.

**Quick start:**

1. **Set up environment**. Install [requirements](https://github.com/supervisely-ecosystem/yolo/blob/master/dev_requirements.txt) manually, or use our pre-built docker image from [DockerHub](https://hub.docker.com/r/supervisely/yolo/tags). Clone [YOLO](https://github.com/supervisely-ecosystem/yolo) repository with model implementation.
2. **Download** your checkpoint from Supervisely Platform.
3. **Run inference**. Refer to our demo scripts: [demo_pytorch.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_pytorch.py), [demo_onnx.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_onnx.py), [demo_tensorrt.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_tensorrt.py)

## Step-by-step guide:

### 1. Set up environment

**Manual installation:**

```bash
git clone https://github.com/supervisely-ecosystem/yolo
cd yolo
pip install -r requirements.txt
```

**Using docker image (advanced):**

We provide a pre-built docker image with all dependencies installed [DockerHub](https://hub.docker.com/r/supervisely/yolo/tags). The image includes installed packages for ONNXRuntime and TensorRT inference.

```bash
docker pull supervisely/yolo:1.0.5
```

See our [Dockerfile](https://github.com/supervisely-ecosystem/yolo/blob/master/docker/Dockerfile) for more details.

Docker image does not include the source code. Clone the repository inside the container:

```bash
git clone https://github.com/supervisely-ecosystem/yolo
```

### 2. Download checkpoint and model files from Supervisely Platform

For YOLO, you need to download only the checkpoint file.

- **For PyTorch inference:** models can be found in the `checkpoints` folder in Team Files after training.
- **For ONNXRuntime and TensorRT inference:** models can be found in the `export` folder in Team Files after training. If you don't see the `export` folder, please ensure that the model was exported to `ONNX` or `TensorRT` format during training.

Go to Team Files in Supervisely Platform and download the files.

![team_files_download](https://github.com/user-attachments/assets/865dea6a-298e-4896-bad9-4066769c0abd)

### 3. Run inference

We provide several demo scripts to run inference with your checkpoint:

- [demo_pytorch.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_pytorch.py) - simple PyTorch inference
- [demo_onnx.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_onnx.py) - ONNXRuntime inference
- [demo_tensorrt.py](https://github.com/supervisely-ecosystem/yolo/blob/master/supervisely_integration/demo/demo_tensorrt.py) - TensorRT inference

# Acknowledgment

This app is based on the `YOLO` model ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)
