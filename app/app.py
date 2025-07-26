import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
YOLOV5_DIR = BASE_DIR / 'yolov5'
MODEL_PATH = BASE_DIR / 'model' / 'best_windows.pt'
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'

# Ensure paths are accessible
sys.path.append(str(YOLOV5_DIR))

# Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'results')

# YOLOv5-specific imports
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.plots import Annotator

# Load model
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(str(MODEL_PATH), device=device)
stride, names = model.stride, model.names
imgsz = 640

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save file
    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Read and preprocess image
    img0 = cv2.imread(save_path)
    if img0 is None:
        return "Invalid image"

    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    label = "No fracture detected"
    annotator = Annotator(img0.copy(), line_width=2, example=str(names))

    if pred[0] is not None and len(pred[0]):
        for *xyxy, conf, cls in reversed(pred[0]):
            label = f'fracture ({conf:.2f})'
            annotator.box_label(xyxy, label, color=(0, 255, 0))  # Green box

    # ✅ Get the final annotated image
    result_image = annotator.result()
        
    result_filename = 'output.png'
    result_img_path = os.path.join('static', 'results', result_filename)

    cv2.imwrite(result_img_path, result_image)

    relative_path = f'results/{result_filename}'
    result_text = "Fracture Detected" if len(pred[0]) else "No fracture detected"
    return render_template("result.html", result_text=result_text, image="results/output.png")


if __name__ == '__main__':
    app.run(debug=True)
