import sys
import time
import cv2
import numpy as np
from pynq_dpu import DpuOverlay

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 inference_dpu.py <dpu_bitfile> <xmodel_file>")
        sys.exit(1)

    bitfile, xmodel = sys.argv[1], sys.argv[2]

    # 1) Load the DPU overlay and model
    overlay = DpuOverlay(bitfile)
    overlay.load_model(xmodel)
    dpu = overlay.runner

    # 2) Read tensor metadata
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()
    fix_point = input_tensors[0].get_attr("fix_point")
    in_dims = tuple(input_tensors[0].dims)     # e.g. (1, 3, 224, 224)
    out_dims = tuple(output_tensors[0].dims)   # e.g. (1, 1000)

    # Prepare reusable buffers
    in_buf  = np.empty(in_dims,  dtype=np.int8)
    out_buf = np.empty(out_dims, dtype=np.int8)

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: unable to open camera.")
        sys.exit(1)
    print("Press 'q' to quit.")

    # Metrics
    frame_count = 0
    total_inference_time = 0.0
    sess_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB, resize to DPU input HxW
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, _, H, W = in_dims
        img = cv2.resize(img, (W, H))
        img = img.astype(np.float32) / 255.0

        # Quantize
        scale = 2 ** fix_point
        img_q = np.round(img * scale).astype(np.int8)

        # Place into buffer (NCHW)
        # img_q is HxWxC, we need CxHxW
        in_buf[0] = img_q.transpose(2, 0, 1)

        # Inference
        t0 = time.time()
        job = dpu.execute_async([in_buf], [out_buf])
        dpu.wait(job)
        t1 = time.time()

        # Compute metrics
        inference_time = t1 - t0
        total_inference_time += inference_time
        frame_count += 1

        # Post‑process: argmax on output
        scores = out_buf[0].astype(np.int32)
        pred = int(scores.argmax())

        # Overlay prediction
        cv2.putText(frame, f"Pred: {pred}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("DPU Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final metrics
    sess_time = time.time() - sess_start
    if frame_count:
        avg_t = total_inference_time / frame_count
        fps = 1.0 / avg_t if avg_t>0 else 0.0
        print(f"\nFrames: {frame_count}")
        print(f"Avg inference time: {avg_t:.4f} s")
        print(f"FPS: {fps:.2f}")
        print(f"Total session time: {sess_time:.2f} s")
    else:
        print("No frames processed.")

if __name__ == "__main__":
    main()
