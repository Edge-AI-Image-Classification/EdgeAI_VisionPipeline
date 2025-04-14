import time
import torch
import torchvision.transforms as tvtransform
import cv2
import sys
import numpy as np
from resnet50 import ResNet50

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

    #metrics
    frame_count = 0
    total_inference_time = 0.0

    start_time = time.time()  # measure total session time (optional)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Time the forward pass
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.argmax(outputs, dim=1).item()
        t1 = time.time()

        inference_time = t1 - t0
        total_inference_time += inference_time
        frame_count += 1

        # Print predicted class (numeric ID)
        label_str = f"Predicted class: {preds}"
        cv2.putText(frame, label_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Edge Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    session_time = time.time() - start_time

    #final metrics
  
    if frame_count > 0:
        avg_inference_time = total_inference_time / frame_count
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

        print(f"\nProcessed {frame_count} frames.")
        print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
        print(f"Approx. FPS: {fps:.2f}")

        # total session time
        print(f"Total session time: {session_time:.2f} seconds")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()
