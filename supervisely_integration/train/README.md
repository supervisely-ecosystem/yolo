<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/ffb1a3b8-0861-4292-9472-96ce6601e8ff"/>  

# Train YOLO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
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

# How to Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/user-attachments/assets/2b5f867e-7d99-4c06-9a4c-96ba29523941" width="100%" style='padding-top: 10px'>  

**Step 2.** Select train / val split

<img src="https://github.com/user-attachments/assets/cb50e8e1-9e10-4195-b000-c694358f5ab1" width="100%" style='padding-top: 10px'>  

**Step 3.** Select the classes you want to train RT-DETRv2 on

<img src="https://github.com/user-attachments/assets/7b5d22a8-6c43-4b3f-b262-01a9ed2e58d4" width="100%" style='padding-top: 10px'>  

**Step 4.** Select the model you want to train

:information_source: The screenshot doesn’t include all the available models. You can view the complete list in the [models.json](../models.json) file.

<img src="https://github.com/user-attachments/assets/fa4df097-4e8d-48ae-b53d-44ff8a7beb5a" width="100%" style='padding-top: 10px'>  

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/user-attachments/assets/4cf73b4e-f3c8-47db-91c6-7145c2352274" width="100%" style='padding-top: 10px'>  

**Step 6.** Enter experiment name and start training

<img src="https://github.com/user-attachments/assets/fdb2943d-7ee2-4ff8-8616-3587911ce270" width="100%" style='padding-top: 10px'>  

**Step 7.** Monitor training progress

<img src="https://github.com/user-attachments/assets/41f2d8b3-dd5a-4f4c-9605-1a3fa5e89827" width="100%" style='padding-top: 10px'>  

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervisely.com/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/10a7a32a-8adc-4cc0-b05d-ff1ef8e08552" width="100%" style='padding-top: 10px'>  

# Acknowledgment

This app is based on the `YOLO` model ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)