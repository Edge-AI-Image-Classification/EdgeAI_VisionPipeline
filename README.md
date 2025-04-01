# ResNet-50 Image Classification & Edge Deployment

A practical pipeline demonstrating:
- **COCO** usage for **image classification** (single label per image)
- **ResNet-50** (CNN-based) for end-to-end training
- **GPU acceleration** (PyTorch, Torchvision, etc.) on PC/server
- **Edge device** inference on resource-constrained hardware (Kria KV260, Jetson Orin NX, Raspberry Pi 5)

With a future expansion to **Faster R-CNN** for full **object detection**.

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Downloading the COCO Dataset](#downloading-the-coco-dataset)  
6. [Usage](#usage)  
   1. [Data Preparation](#data-preparation)  
   2. [Model Training](#model-training)  
   3. [Evaluation](#evaluation)  
   4. [Inference on Edge Devices](#inference-on-edge-devices)  
7. [Results](#results)  
8. [Future Expansion: Faster R-CNN Object Detection](#future-expansion-faster-r-cnn-object-detection)  
9. [Troubleshooting](#troubleshooting)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [References](#references)

---

## Overview

This repository, **EdgeAI-Classification-pipeline**, demonstrates **single-label image classification** using **ResNet-50** trained on a **COCO-based dataset**. Although COCO is typically used for object detection, here we **treat each image as having one main label**, then train a ResNet classifier. After training, we **export** the model for deployment on edge devices like the **NVIDIA Jetson Orin NX**, **Xilinx Kria KV260**, or **Raspberry Pi 5**. This allows **live camera inference** where the model classifies the dominant object in the scene.

We also include a **Future Expansion** section on how to evolve this pipeline into a **Faster R-CNN**–based **object detection** system (with bounding boxes).

---

## Features

- **COCO** – Adapting a subset or a filtered version of the 80 classes for single-label classification.  
- **ResNet-50** – A standard CNN architecture for image classification.  
- **GPU Training** – Leverage PyTorch + CUDA for faster training on a desktop/server.  
- **Edge Inference** – Minimal script to run classification on a camera feed, labeling each frame with a single class.  
- **Future Detectors** – Stretch goal to use **Faster R-CNN** or other object detection models for bounding boxes.

---

## Project Structure

A typical layout might look like this:

```
.
├── coco2017/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── instances_train2017.json
│       ├── instances_val2017.json
│       ...
├── src/
│   ├── train_resnet.py      # Main training script for ResNet-50 classification
│   ├── infer_edge.py        # Script to run real-time camera inference
│   └── ...
├── README.md
└── requirements.txt
```

Feel free to reorganize as needed.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/EdgeAI_VisionPipeline.git
cd EdgeAI-VisionPipeline
```

### 2. Install Dependencies

```bash
pip install -r dependencies.txt
```

*(If you’re using Conda, you can instead create a conda environment and install PyTorch + Torchvision per your GPU drivers.)*

---

## Downloading the COCO Dataset

1. **Create** a directory for COCO (e.g., `coco2017`) if you haven’t already.  
2. **Download** train and val images plus annotations from the [COCO download page](https://cocodataset.org/#download). For example (Linux/WSL):
   ```bash
   mkdir coco2017
   cd coco2017
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```
3. **Unzip** the files:
   ```bash
   unzip train2017.zip
   unzip val2017.zip
   unzip annotations_trainval2017.zip
   ```
4. **Check** that your structure is:
   ```
   coco2017/
   ├── train2017/
   ├── val2017/
   └── annotations/
       ├── instances_train2017.json
       ├── instances_val2017.json
       ...
   ```

---

## Usage

### Data Preparation

By default, **COCO** is structured for **object detection** (multiple bounding boxes). To create a **single-label classification** dataset, you might:
- Filter images to keep only **one** dominant object category per image, or  
- Assign each image a **primary** category if multiple exist, or  
- Use a smaller custom subset (e.g., “pizza,” “hot dog,” etc.).

Your `train_resnet.py` can parse COCO annotations (via `pycocotools`) but only keep those images with a single bounding box of a target category. That bounding box category is your **label**.

### Model Training

```bash
python src/train_resnet.py
```

- Loads COCO from `coco2017/` (with your single-label logic).  
- Builds a **ResNet-50** classifier, possibly **fine-tuning** from ImageNet.  
- **Trains** on GPU if available, logs loss, etc.  
- Saves the final model to `resnet_coco.pth`.

### Evaluation

Depending on your script:
- Check standard classification metrics (accuracy, confusion matrix) on a validation split.  
- Inspect predictions on a handful of images.

### Inference on Edge Devices

After training:

1. **Copy** `resnet_coco.pth` (the saved model) to the edge device.
2. **Install** the appropriate Python environment (PyTorch or ONNX runtime, plus OpenCV).
3. **Run**:

```bash
python src/infer_edge.py resnet_coco.pth
```

The script should:
- Open a camera feed,  
- Preprocess each frame for ResNet-50,  
- Run **single-label classification**,  
- Overlay the predicted class on the live video feed.

*(Keep in mind a single-label classifier might misclassify if multiple objects are visible. This approach is best for “one main object” scenes.)*

---

## Results

- **Accuracy** on your custom-labeled COCO subset will vary widely depending on how you filter categories and how many images you keep. Typical classification accuracy might be 70–90%+ for well-defined classes.  
- **Real-Time Inference**: 
  - Desktop GPU can easily achieve 20+ FPS for classification.  
  - On **Jetson Orin NX**, **Kria KV260**, or **Raspberry Pi 5**, you might get a few to 10+ FPS depending on optimization (e.g., TensorRT, quantization).  
- **Optimizations**: If speed is insufficient, consider a smaller backbone (MobileNet), or hardware accelerators.

---

## Future Expansion: Faster R-CNN Object Detection

If you need **multiple objects** or **bounding boxes** within the same frame, a **vanilla ResNet** classifier won’t suffice. You could extend this project to **Faster R-CNN** (or YOLO, SSD, etc.), which leverages:

- A **ResNet-50 (or deeper)** backbone,
- A **Region Proposal Network (RPN)**,
- **Bounding box regression** and **classification heads** for each region.

This would let you detect (and label) all objects in the frame, not just classify the entire image. The main changes would be:

1. **Parsing** COCO detection annotations to keep bounding boxes and labels for each image.  
2. **Using** `torchvision.models.detection.fasterrcnn_resnet50_fpn()` or a similar model in PyTorch.  
3. **Training** with the standard detection losses (e.g., box regression, classifier, RPN).  
4. **Inference** includes bounding-box drawing on the camera feed, rather than a single label.

We plan to add a dedicated **Faster R-CNN** training script to this repo in the future, so stay tuned!

---

## Troubleshooting

1. **Dataset Paths**  
   - Ensure `coco2017/` paths match your script’s expectations.  
2. **CUDA Device**  
   - If you don’t have an NVIDIA GPU or correct drivers, training will fall back to CPU. This can be slow for large datasets.  
3. **Single vs. Multi Objects**  
   - If your frames contain multiple objects, a single-label classifier might be confused. Consider object detection for multi-object scenes.  
4. **Inference Speed**  
   - For edge devices, use smaller input sizes or optimization frameworks (e.g., TensorRT, OpenVINO) to speed up classification.

---

## Contributing

1. **Fork** this repository.  
2. **Create** a new branch for your feature/fix:
   ```bash
   git checkout -b feature-my-improvement
   ```
3. **Commit** your changes and push to your fork:
   ```bash
   git commit -m "Add my new feature"
   git push origin feature-my-improvement
   ```
4. **Open a Pull Request** into the main branch.

We welcome community contributions, bug reports, and suggestions!

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute the code as allowed by that license.

---

## References

1. **COCO Dataset** – [Official Site](https://cocodataset.org/)  
2. **PyTorch Torchvision** – [Docs](https://pytorch.org/vision/stable/index.html)  
3. **ResNet Paper** – [Deep Residual Learning for Image Recognition (He et al.)](https://arxiv.org/abs/1512.03385)  
4. **NVIDIA Jetson** – [Developer Site](https://developer.nvidia.com/embedded-computing)  
5. **Kria KV260** – [Xilinx Documentation](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)  
6. **Raspberry Pi** – [Official Site](https://www.raspberrypi.com/)  

---

_Thank you for visiting **EdgeAI-Classification-pipeline**! If you have any questions, suggestions, or issues, feel free to [open an issue](../../issues) or reach out._
