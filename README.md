# ResNet-50 Image Classification & Edge Deployment

A practical pipeline demonstrating:

- **Caltech101** usage for **single-label image classification**  
- **ResNet-50** (CNN-based) for end-to-end training  
- **GPU acceleration** (PyTorch, Torchvision, etc.) on PC/server  
- **Edge device inference** on resource-constrained hardware (Kria KV260, Jetson Orin NX, Raspberry Pi 5)  

**No separate dataset download required** – the code automatically fetches Caltech101 if it’s not already present.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
   - [Creating & Activating a Conda Environment](#creating--activating-a-conda-environment)  
   - [Installing Dependencies](#installing-dependencies)  
5. [Usage](#usage)  
   - [Data Preparation](#data-preparation)  
   - [Model Training](#model-training)  
   - [Evaluation](#evaluation)  
   - [Inference on Edge Devices](#inference-on-edge-devices)  
6. [Results](#results)  
7. [Troubleshooting](#troubleshooting)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [References](#references)

---

## Overview

This repository, **EdgeAI-Classification-pipeline**, demonstrates **ResNet-50** classification on **Caltech101**. The code automatically **downloads** Caltech101 through the PyTorch `torchvision.datasets.Caltech101` class if it’s not already in your `data/` folder. After training, the project saves a checkpoint (`resnet.pth`) that you can copy to various edge devices (e.g., Jetson, Kria, Raspberry Pi) to perform real-time **camera inference**.

---

## Features

- **Automatic Dataset Download**: No manual steps; if Caltech101 is absent, PyTorch downloads it.  
- **ResNet-50**: A CNN architecture with 102 output classes (101 categories + background).  
- **Single-Label Classification**: One predicted label per image.  
- **GPU or CPU**: By default, code can run on CPU or CUDA if available.  
- **Edge Inference**: Provided script for real-time camera classification on resource-constrained devices.

---

## Project Structure

You have **seven** main Python files:

```
.
├── dataPreprocessLoader.py   # Loads and splits Caltech101; auto-downloads if needed
├── modelTraining.py          # Main training loop for ResNet-50
├── inference_edge.py         # Real-time camera inference on edge devices
├── validation.py             # Evaluation function for validation data
├── resnet50.py               # Defines custom ResNet-50 model
├── residual_block.py         # Defines the building blocks for ResNet
├── plots.py                  # Hardcoded plotting utility (loss/accuracy)
└── dependencies.txt          # Dependencies needed for this project
```

## Installation

### Creating & Activating a Conda Environment

1. **Install** [Anaconda/Miniconda](https://docs.conda.io/en/latest/) if you haven’t already.  
2. **Create** a new environment (example name: `edgeai`):  
   ```bash
   conda create -n edgeai python=3.9
   ```
3. **Activate** it:  
   ```bash
   conda activate edgeai
   ```

### Installing Dependencies

Within your **activated** conda environment:

```bash
conda install --file dependencies.txt
```

Or, if you prefer `pip`:

```bash
pip install -r dependencies.txt
```

*(Make sure you are in the conda environment so packages are installed there.)*

> **GPU/CUDA Note**  
> If you’re running on **GPU** and want CUDA acceleration, ensure your **PyTorch** install is compatible with your installed CUDA driver. You might do:
> ```bash
> conda install pytorch torchvision cudatoolkit=<version> -c pytorch
> ```
> or install a **compatible wheel** (e.g., from PyPI) that matches your CUDA version. Installing via `pip install -r dependencies.txt` alone may not set up CUDA support automatically. You also need to have the appropriate **NVIDIA drivers** and **CUDA toolkit** for your system.

**Typical Requirements** (already in `dependencies.txt`):
- Python 3.9 (or similar)
- PyTorch (with `torchvision`)
- OpenCV (e.g., `opencv-python`)
- Matplotlib (for plotting)
- scikit-learn (for classification reports)
- Others if needed (e.g., TQDM, etc.)

---

## Usage

### 1. Data Preparation

**No manual download** required. When you run `dataPreprocessLoader.py`, the **Caltech101** dataset is automatically fetched and stored in a `data/` folder inside your working directory:

```python
dataset = Caltech101(root='data', download=True, transform=transform)
```

This script splits the dataset into **train** and **validation** sets (80/20) and creates PyTorch `DataLoader` objects (`trainDataloader`, `valDataloader`).

### 2. Model Training

Run:

```bash
python modelTraining.py
```

What happens:

1. **Imports** `trainDataloader` and `valDataloader` from `dataPreprocessLoader.py`.  
2. Builds a **ResNet-50** with 102 output classes (`resnet50.py`).  
3. Trains on CPU or CUDA (change `device = torch.device("cpu")` to `torch.device("cuda")` if you want GPU).  
4. Prints training/validation loss and accuracy each epoch.  
5. **Saves** the final model to `resnet.pth`.

### 3. Evaluation

The training code calls `evaluate(...)` from `validation.py` after each epoch to compute validation loss and accuracy. You can also modify `validation.py` to produce confusion matrices or other metrics.

### 4. Inference on Edge Devices

1. **Copy** `resnet.pth` to your device (e.g., `scp` to Jetson).  
2. Install **PyTorch** (CPU or GPU version) and **OpenCV** on that device.  
3. Run:
   ```bash
   python inference_edge.py resnet.pth
   ```
4. This opens your default camera (ID=0), preprocesses each frame to 224×224, and feeds it into the model. The script prints and overlays the numeric class ID. *(You can add a label mapping if desired.)*

---

## Results

- The project logs **training** and **validation** accuracy for each epoch.  
- Caltech101 is relatively small, so you can train quickly on CPU or GPU.  
- If you want to visualize metrics, see the **plots.py** script (though it currently has hardcoded data).

---

## Troubleshooting

1. **No GPU Detected**  
   - By default, `modelTraining.py` sets `device = torch.device("cpu")`. Change to `torch.device("cuda")` if you have a supported GPU.  
2. **Dataset Download Failures**  
   - If your internet connection is restricted or the download fails, ensure you can reach the PyTorch data mirrors or manually place the dataset in the `data/` folder.  
3. **Edge Device Issues**  
   - For Jetson/ARM boards, install PyTorch specifically built for that architecture.  

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

We welcome suggestions, bug reports, and community contributions!

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute the code as allowed by that license.

---

## References

1. **Caltech101 Dataset** – [PyTorch Docs (Caltech101)](https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech101.html)  
2. **ResNet Paper** – [Deep Residual Learning for Image Recognition (He et al.)](https://arxiv.org/abs/1512.03385)  
3. **PyTorch** – [Docs](https://pytorch.org/)  
4. **NVIDIA Jetson** – [Developer Site](https://developer.nvidia.com/embedded-computing)  
5. **Xilinx Kria KV260** – [Documentation](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)  
6. **Raspberry Pi** – [Official Site](https://www.raspberrypi.com/)

---

_Thank you for visiting **EdgeAI-Classification-pipeline**! If you have any questions or issues, feel free to open an [issue](../../issues) or reach out._
