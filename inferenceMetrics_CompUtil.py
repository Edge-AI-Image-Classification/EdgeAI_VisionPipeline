import time
import torch
import torchvision.transforms as tvtransform
import cv2
import sys
import numpy as np
from resnet50 import ResNet50

##############################
# 1) TEGRASTATS (Jetson Orin)
##############################
import subprocess
import re

def get_tegrastats_snapshot():
    """
    Returns a dict with CPU/GPU usage from a single tegrastats call.
    This only works on NVIDIA Jetson devices with tegrastats installed.
    """
    try:
        output = subprocess.check_output(["tegrastats", "--interval", "100", "--once"])
        line = output.decode("utf-8").strip()

        #CPU and GPU usage
        # CPU usage
        cpu_match = re.search(r"cpu \[(.*?)\]", line)
        cpu_usage = None
        if cpu_match:
            usage_str = cpu_match.group(1)  # e.g. "4%,0%,3%,1%"
            usage_vals = [float(val.strip().replace("%","")) for val in usage_str.split(",")]
            # average CPU usage across cores
            cpu_usage = sum(usage_vals) / len(usage_vals)

        # GPU usage (GR3D_FREQ)
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        gpu_usage = float(gpu_match.group(1)) if gpu_match else None

        return {"cpu_usage": cpu_usage, "gpu_usage": gpu_usage}
    except Exception as e:
        print(f"[TEGRASTATS ERROR] {e}")
        return {"cpu_usage": None, "gpu_usage": None}

# 2) PSUTIL (Raspberry Pi)
# install psutil with before running pip install psutil
try:
    import psutil
    def get_psutil_stats():
        """
        Returns CPU usage percent (overall) and memory usage in MB for this system.
        """
        cpu_percent = psutil.cpu_percent(interval=None)  # snapshot
        mem_info = psutil.virtual_memory()
        mem_used_mb = mem_info.used / (1024 * 1024)
        return {"cpu_percent": cpu_percent, "mem_used_mb": mem_used_mb}
except ImportError:
    print("psutil not installed. CPU/mem usage for Pi or generic Linux won't be available.")
    def get_psutil_stats():
        return {"cpu_percent": None, "mem_used_mb": None}

#3) XILINX KRIA KV260 (xbutil examine placeholder)
def get_xbutil_stats():
    """
    Placeholder for capturing usage from a Kria KV260 if XRT is installed.
    Command might differ based on OS or XRT version. 
    Example (not guaranteed to work out-of-the-box):
        xbutil examine -r usage
    """
    try:
        cmd = ["xbutil", "examine", "-r", "usage"]  # or "xbutil examine -r thermal" etc.
        output = subprocess.check_output(cmd).decode("utf-8")
        # Example parse. Actual output might differ. 
        # We'll just return the raw text or do a minimal parse.
        return {"xbutil_output": output}
    except Exception as e:
        print(f"[XBUTIL ERROR] {e}")
        return {"xbutil_output": None}


def load_model(model_path, device):
    model = ResNet50(num_classes=102)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inference_edge.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = tvtransform.Compose([
        tvtransform.ToPILImage(),
        tvtransform.Resize((224, 224)),
        tvtransform.ToTensor(),
        tvtransform.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    print("Press 'q' to quit.")

    # metrics
    frame_count = 0
    total_inference_time = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -- Measure device stats (comment/uncomment based on your hardware) --

        # 1) Jetson tegrastats
        #stats_jetson = get_tegrastats_snapshot()
        #if stats_jetson["cpu_usage"] is not None:
        #    print(f"[JETSON] CPU Usage: {stats_jetson['cpu_usage']:.2f}%, GPU Usage: {stats_jetson['gpu_usage']:.2f}%")

        # 2) Raspberry Pi or general Linux psutil
        #stats_psutil = get_psutil_stats()
        #if stats_psutil["cpu_percent"] is not None:
        #    print(f"[PSUTIL] CPU: {stats_psutil['cpu_percent']:.1f}%, Mem: {stats_psutil['mem_used_mb']:.1f} MB")

        # 3) Xilinx Kria KV260 (placeholder using xbutil)
        #stats_xbutil = get_xbutil_stats()
        #if stats_xbutil["xbutil_output"] is not None:
        #    # For demonstration, just print the raw text or parse as needed
        #    print("[XBUTIL] Usage info:")
        #    print(stats_xbutil["xbutil_output"])

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # time the inference
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.argmax(outputs, dim=1).item()
        t1 = time.time()

        inference_time = t1 - t0
        total_inference_time += inference_time
        frame_count += 1

        label_str = f"Predicted class: {preds}"
        cv2.putText(frame, label_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Edge Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    session_time = time.time() - start_time

    # final metrics
    if frame_count > 0:
        avg_inference_time = total_inference_time / frame_count
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

        print(f"\nProcessed {frame_count} frames.")
        print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
        print(f"Approx. FPS: {fps:.2f}")
        print(f"Total session time: {session_time:.2f} seconds")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()
