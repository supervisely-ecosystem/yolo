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
