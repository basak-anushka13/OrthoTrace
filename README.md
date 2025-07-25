OrthoTrace 🧴 - Fracture Detection from X-ray Images using YOLOv5

OrthoTrace is a web application that allows users to upload X-ray images and automatically detects bone fractures using a YOLOv5 deep learning model.

🚀 Features

Upload any X-ray image (wrist, elbow, etc.)

Detects and highlights fractures with bounding boxes

Displays the result in a web interface

Built using Flask, OpenCV, PyTorch, and YOLOv5

🔧 Project Structure

OrthoTrace/
├── app/
│   ├── app.py               # Flask app
│   ├── templates/
│   │   ├── index.html       # Upload page
│   │   └── result.html      # Result display page
│   ├── static/
│   │   ├── uploads/         # User uploads
│   │   └── results/         # Annotated images
│   └── model/
│       └── best_windows.pt  # Trained YOLOv5 model
├── yolov5/                  # YOLOv5 directory (from Ultralytics)
├── requirements.txt
└── README.md

🖼️ Demo

🔗 Live Demo: (To be deployed — currently run locally using Flask)

To run locally:

# Clone this repo
git clone https://github.com/basak-anushka13/OrthoTrace.git
cd OrthoTrace

# Set up Python virtual environment
conda activate fractureenv  # or python -m venv env
pip install -r requirements.txt

# Run the app
cd app
python app.py

Then visit: http://127.0.0.1:5000

📦 Requirements

Install all dependencies using:

pip install -r requirements.txt

💡 Model Info

Model: YOLOv5 custom trained on fracture detection data

File: best_windows.pt

Format: PyTorch .pt model

✍️ Credits

Built by Anushka & Team, 2025

Based on Ultralytics YOLOv5

📌 Future Scope

Multi-fracture detection

Integration with patient management systems

Deployment on cloud (Render / HuggingFace / Streamlit Cloud)
