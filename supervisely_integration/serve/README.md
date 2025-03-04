<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/f1fffbf3-bfd1-474e-8826-d0295c403946"/>  

# Serve YOLO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolo/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolo)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/yolo/supervisely_integration/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/yolo/supervisely_integration/serve.png)](https://supervisely.com)

</div>

# Overview

Serve different checkpoints from YOLO architecures as a Supervisely Application. YOLO is a powerful neural network architecture that provides both decent accuracy of predictions and high speed of inference.

You can deploy models in optimized runtimes:

- **TensorRT** is a very optimized environment for Nvidia GPU devices. TensorRT can significantly boost the inference speed. Additionally, you can select *FP16 mode* to reduce GPU memory usage and further increase speed. Usually, the accuracy of predictions remains the same.
- **ONNXRuntime** can speed up inference on some CPU and GPU devices.

# How to Run

0. Start the application from an app's context menu or the Ecosystem.

1. Select pre-trained model or custom model trained inside Supervisely platform, and a runtime for inference.

<img src="https://github.com/user-attachments/assets/e5969562-42ab-4ca7-a070-fc1db17ed4e5" />

2. Select device and press the `Serve` button, then wait for the model to deploy.

<img src="https://github.com/user-attachments/assets/37ed45b7-f72b-4789-b446-a80e5f419219" />

3. You will see a message once the model has been successfully deployed.

<img src="https://github.com/user-attachments/assets/032b3767-f190-4dd2-bed8-67e6c746625a" />

4. You can now use the model for inference and see model info.

<img src="https://github.com/user-attachments/assets/d6d212ac-4b52-4744-ae74-ac15800e55a2" />

# Acknowledgment

This app is based on the `YOLO` model ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)
