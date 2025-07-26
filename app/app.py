from flask import Flask, render_template, request, redirect, url_for
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Add yolov5 directory to Python path
FILE = Path(__file__).resolve()
YOLOV5_DIR = FILE.parent / 'yolov5'
sys.path.append(str(YOLOV5_DIR))

from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator
from models.common import DetectMultiBackend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv5 model
device = select_device('')
weights = str(Path(__file__).parent / "model" / "best_windows.pt")
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            print("❌ No file part in the request")
            return "No file part in the request", 400

        file = request.files['image']
        if file.filename == '':
            print("❌ No file selected")
            return "No selected file", 400

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"📂 File saved to: {filepath}")

        # Load and preprocess
        image = Image.open(filepath).convert('RGB')
        img0 = np.array(image)
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.4)

        # Annotate
        result_img = img0.copy()
        annotator = Annotator(result_img, line_width=2)
        for i, det in enumerate(pred):
            print(f"🔍 Detections: {len(det)}")
            if len(det):
                det[:, :4] = det[:, :4].round()
                for *xyxy, conf, cls in reversed(det):
                    label = "Fracture"
                    annotator.box_label(xyxy, label, color=(0, 255, 0))

        # Save result
        result_filename = "result_" + file.filename
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        Image.fromarray(cv2.cvtColor(annotator.result(), cv2.COLOR_BGR2RGB)).save(result_path)
        print(f"✅ Result saved to: {result_path}")

        return render_template("result.html", result_image=result_filename)

    except Exception as e:
        print("🔥 Error during prediction:", e)
        return "An internal error occurred.", 500

if __name__ == '__main__':
    app.run(debug=True)
