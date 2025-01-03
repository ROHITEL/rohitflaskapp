from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Dynamically set the base directory for accessing model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure upload folder exists in the deployment environment
UPLOAD_FOLDER = '/tmp'  # Use '/tmp' for Render compatibility
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLOv4 model and configuration
net = cv2.dnn.readNet(
    os.path.join(BASE_DIR, "yolov4-tiny.weights"),
    os.path.join(BASE_DIR, "yolov4-tiny.cfg")
)

# Load the COCO class labels
with open(os.path.join(BASE_DIR, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/')
def index():
    return render_template('website.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file to a temporary location
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Process the image using YOLO
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Unable to read the uploaded image'})

        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        result = []
        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                result.append({'label': label, 'confidence': confidence, 'bbox': [x, y, w, h]})

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


