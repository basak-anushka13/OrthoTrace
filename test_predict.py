import torch
import cv2
import os
from pathlib import Path
import sys
import numpy as np

# ✅ Set paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"  # yolov5 folder
sys.path.append(str(ROOT))

# ✅ Imports from YOLOv5
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator

# ✅ Set image path
image_path = 'app/static/uploads/img1.jpg'
assert os.path.exists(image_path), f"Image not found: {image_path}"

# ✅ Load model
weights = 'app/model/best_windows.pt'
device = select_device('')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)

# ✅ Read and preprocess image
img0 = cv2.imread(image_path)
assert img0 is not None, f"Failed to load {image_path}"
img = letterbox(img0, imgsz, stride=stride)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
img = np.ascontiguousarray(img)

# ✅ Prepare tensor
img_tensor = torch.from_numpy(img).to(device).float()
img_tensor /= 255.0
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)

# ✅ Inference
pred = model(img_tensor, augment=False, visualize=False)
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

# ✅ Draw boxes using Annotator
for i, det in enumerate(pred):
    annotator = Annotator(img0.copy(), line_width=2)
    if len(det):
        det[:, :4] = det[:, :4].round()  # boxes already scaled
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=(0, 255, 0))

    result_img = annotator.result()

# ✅ Save result
output_path = 'runs/test_results/result.jpg'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, result_img)
print(f"\n✅ Prediction complete. Output saved to: {output_path}")

# ✅ OPTIONAL: Display image using OpenCV (this pops up a window)
cv2.imshow("Prediction", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
