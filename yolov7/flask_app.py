import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from inference_onnx import model_inference, post_process
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent

app = Flask(__name__)

UPLOAD_FOLDER = './user_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PREDICTIONS_FOLDER = './predictions'
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'})

    file = request.files['image']

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the saved image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform prediction
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) / 255
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)

    model_onnx_path = os.path.join(BASE_DIR, "yolov7-p6-bonefracture.onnx")
    output = model_inference(model_onnx_path, img)
    out_img, out_txt = post_process(file_path, output)
    out_img = out_img[..., ::-1]

    # Save the predicted image
    predicted_image_path = os.path.join(app.config['PREDICTIONS_FOLDER'], f'{file.filename}')
    cv2.imwrite(predicted_image_path, out_img)

    return jsonify({'predicted_image': predicted_image_path})

if __name__ == '__main__':
    app.run()
