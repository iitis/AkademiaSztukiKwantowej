#!/usr/bin/env python
# maskrcnn_demo.py --------------------------------------------------
"""
Minimal Mask-R-CNN demo (instance segmentation).
$ python maskrcnn_demo.py [optional_image.jpg]
Outputs: maskrcnn_out.jpg with coloured masks, boxes & labels.
"""
# ------------------------------------------------------------------
import sys, time, urllib.request, random, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models, utils

# ------------ setup ------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model   = models.detection.maskrcnn_resnet50_fpn(weights=weights) \
                       .to(DEVICE).eval()
CATS    = weights.meta["categories"]

# ------------ pick an image ---------------------------------------
if len(sys.argv) > 1:
    img_path = Path(sys.argv[1])
else:                                                # fallback: COCO sample
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img_path = Path("sample.jpg")
    if not img_path.exists():
        urllib.request.urlretrieve(url, img_path)

img = Image.open(img_path).convert("RGB")

# ------------ preprocessing ----------------------------------------
tfm   = transforms.Compose([transforms.ToTensor()])  # (0-1) tensor
tensor = tfm(img).to(DEVICE)                         # (3,H,W)

# ------------ inference -------------------------------------------
with torch.no_grad():
    t0 = time.perf_counter()
    preds = model([tensor])[0]                       # list[0]
    dt  = time.perf_counter() - t0

# keep instances with score > 0.5 --------------------
keep = preds["scores"] > 0.5
boxes = preds["boxes"][keep]
labels= preds["labels"][keep]
masks = preds["masks"][keep] > 0.5                   # (N,1,H,W) → bool

# ------------ draw masks & boxes -------------------------------
if len(masks):
    # random pastel colours
    colours = [
        tuple(random.randint(100,255) for _ in range(3)) for _ in range(len(masks))
    ]
    drawn = utils.draw_segmentation_masks(
        (tensor.cpu()*255).byte(),                   # CHW byte image
        masks.squeeze(1).cpu(),                      # (N,H,W) bool
        alpha=0.6,
        colors=colours)
    drawn = utils.draw_bounding_boxes(
        drawn, boxes.cpu().round().int(),
        [CATS[i] for i in labels.cpu()],
        colors=colours, width=2)
    out_img = Image.fromarray(drawn.permute(1,2,0).numpy())
else:                                                # nothing above threshold
    out_img = img.copy()

out_file = img_path.parent / "maskrcnn_out.jpg"
out_img.save(out_file)

# ------------ console report --------------------------------------
print(f"\nMask-R-CNN detections for {img_path.name} "
      f"({DEVICE}, {dt*1e3:.1f} ms, score>0.5):")
for box, lab, conf in zip(boxes, labels, preds["scores"][keep]):
    x1,y1,x2,y2 = box.tolist()
    print(f"  {CATS[lab]:15s} {conf*100:5.1f}%  "
          f"({x1:4.0f},{y1:4.0f}) – ({x2:4.0f},{y2:4.0f})")

print(f"\nSaved annotated image to {out_file.resolve()}")
