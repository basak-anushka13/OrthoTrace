import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

FILE = Path(__file__).resolve()
ROOT = FILE.parent / 'yolov5'  # correct: same level as app.py
sys.path.append(str(ROOT))

# ✅ Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load model once
device = select_device('')
weights = r'C:\Users\OTHERS\Desktop\OrthoTrace\app\model\best_windows.pt'  # full path to .pt
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['xray']
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # ✅ Read and preprocess image
        img0 = cv2.imread(file_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # ✅ Model prediction
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.4)

        # ✅ Annotate results
        annotator = Annotator(img0.copy(), line_width=2)
        for det in pred:
            print("Detections found:", len(det))
            if len(det):
                det[:, :4] = det[:, :4].round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'Fracture {conf:.2f}'
                    annotator.box_label(xyxy, label, color=(0, 255, 0))

        result_img = annotator.result()
        result_path = os.path.join(RESULT_FOLDER, filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, result_img)

        return render_template('result.html', result_img=result_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
