#!/usr/bin/env python
# resnet_demo.py ----------------------------------------------------
"""
Minimal, laptop-friendly ResNet-18 ImageNet demo.
$ python resnet_demo.py [optional_image.jpg]
"""
# -------------------------------------------------------------------
import sys, urllib.request, time, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
# ------------ setup ------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.DEFAULT          # ← built-in weights
MODEL   = models.resnet18(weights=weights).to(DEVICE).eval()

CATEGORIES = weights.meta["categories"]            # 1 000 class names

# ------------ pick an image ---------------------------------------
if len(sys.argv) > 1:                              # user-supplied file
    img_path = Path(sys.argv[1])
else:                                              # fallback sample
    from torchvision.datasets.utils import download_url
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img_path = Path("dog.jpg")
    if not img_path.exists():
        download_url(url, ".", filename="sample.jpg")#, quiet=True)

img = Image.open(img_path).convert("RGB")

# ------------ preprocessing ----------------------------------------
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]),
])
batch = tfm(img).unsqueeze(0).to(DEVICE)           # (1,3,224,224)

# ------------ inference --------------------------------------------
with torch.no_grad():
    t0 = time.perf_counter()
    logits = MODEL(batch)
    dt = time.perf_counter() - t0

prob = torch.softmax(logits, dim=1)[0]
top5 = prob.topk(5)

# ------------ results ---------------------------------------------
print(f"\nTop-5 predictions for {img_path.name}  (ResNet-18, {DEVICE}, "
      f"{dt*1e3:.1f} ms):")
for rank, (idx, p) in enumerate(zip(top5.indices, top5.values), 1):
    print(f"{rank}. {CATEGORIES[idx]} – {p.item()*100:.1f}%")
