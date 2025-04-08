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

    #defining transformations (should match training preprocessing)
    transform = tvtransform.Compose([
        tvtransform.ToPILImage(),
        tvtransform.Resize((224, 224)),
        tvtransform.ToTensor(),
        tvtransform.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    #opening camera on default port w/ ID=0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.argmax(outputs, dim=1)
            pred_class = preds.item()
        print(f"Predicted class: {pred_class}")

        #displaying classes as number ID
        label_str = f"Class ID: {pred_class}"
        cv2.putText(frame, label_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Edge Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
