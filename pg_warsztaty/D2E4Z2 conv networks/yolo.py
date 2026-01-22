#!/usr/bin/env python
# yolo_demo.py ------------------------------------------------------
"""
Minimal YOLOv8-nano object-detection demo.
$ python yolo_demo.py [optional_image.jpg]
Produces: yolo_out.jpg with boxes + labels + scores.
"""
# ------------------------------------------------------------------
import sys, time, urllib.request, torch
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO          # pip install ultralytics

# ------------ setup ------------------------------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"
model  = YOLO("yolov8n.pt")           # nano = fastest, laptop-friendly

# ------------ pick an image ---------------------------------------
if len(sys.argv) > 1:
    img_path = Path(sys.argv[1])
else:  # fallback sample (street scene, ≈60 kB)
    url      = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    img_path = Path("sample.jpg")
    if not img_path.exists():
        urllib.request.urlretrieve(url, img_path)

img = Image.open(img_path).convert("RGB")          # PIL image

# ------------ inference -------------------------------------------
t0 = time.perf_counter()
results = model.predict(img, device=DEVICE, verbose=False)  # list[0]
dt = time.perf_counter() - t0
res = results[0]                                           # first image

# ------------ draw & save -----------------------------------------
annotated = res.plot()                     # → BGR numpy array with boxes
out_file  = img_path.parent / "yolo_out.jpg"
Image.fromarray(annotated[..., ::-1]).save(out_file)

# ------------ console report --------------------------------------
print(f"\nDetections for {img_path.name} (YOLO-v8n, device={DEVICE}, "
      f"{dt*1e3:.1f} ms):")
for cls, conf, box in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
    label = model.model.names[int(cls)]
    x1, y1, x2, y2 = box.tolist()
    print(f"  {label:12s} {conf*100:5.1f}%  "
          f"({x1:4.0f},{y1:4.0f}) – ({x2:4.0f},{y2:4.0f})")

print(f"\nSaved annotated image to {out_file.resolve()}")
