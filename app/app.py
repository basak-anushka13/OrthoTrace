import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

# --- Base directories ---
BASE_DIR = Path(__file__).resolve().parent  # app/
YOLOV5_DIR = BASE_DIR.parent / 'yolov5'     # ../yolov5
MODEL_PATH = BASE_DIR / 'model' / 'best_windows.pt'
UPLOAD_FOLDER = BASE_DIR / 'static' / 'results'

# --- Add yolov5 to Python path ---
sys.path.append(str(YOLOV5_DIR))

# --- Flask App ---
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# --- YOLOv5 Imports (after sys.path) ---
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.plots import Annotator

# Load model
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(r"C:\Users\OTHERS\Desktop\OrthoTrace\app\model\best_windows.pt", device=device)
print("Model loaded:", MODEL_PATH)
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

    if file:
        # Save uploaded image
        filename = file.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        # --- Load image and preprocess ---
        device = select_device('')
        model = DetectMultiBackend(str(MODEL_PATH), device=device)
        stride, pt = model.stride, model.pt
        names = model.names

        img0 = cv2.imread(img_path)  # BGR
        img = letterbox(img0, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # --- Run detection ---
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False)

        # --- Annotate image ---
        annotator = Annotator(img0.copy(), line_width=3)
        found = False
        for det in pred:
            if len(det):
                found = True
                for *xyxy, conf, cls in reversed(det):
                    label = f'Fracture {conf:.2f}'
                    annotator.box_label(xyxy, label, color=(0, 255, 0))

        result_img = annotator.result()
        result_filename = 'output_' + filename
        result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_img_path, result_img)

        # --- Prepare result ---
        result_text = "Fracture detected!" if found else "No fracture detected."
        return render_template("result.html", result_text=result_text, image=f"results/{result_filename}")


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
