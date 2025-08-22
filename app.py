from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from torchvision import transforms
import timm
from src.kidneyDiseaseClassifier import logger
import os
from kidneyDiseaseClassifier.utils.common import decodeImage
from kidneyDiseaseClassifier.pipeline.prediction import PredictionPipeline
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

allowed_origins = ['https://meta-doctor.vercel.app', 'https://meta-doctor-git-main-priyanshumalaviya9210-gmailcom.vercel.app', 'https://meta-doctor-nwxeblfpx-priyanshumalaviya9210-gmailcom.vercel.app', 'https://oral-disease-classification.vercel.app', 'http://localhost:3000', 'http://127.0.0.1:3000',
                   'https://oral-disease-classification.vercel.app/']


# Function to load the model (modify according to your needs)
def load_model(model_name, num_classes, device):
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes)
    model_path = 'best_model/efficientvit_m2_kidney_disease_classifier.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@app.route('/predict', methods=['POST'])
@cross_origin(origins=allowed_origins)
def predict_image():
    try:
        data = request.json
        image_data = data['image']

        # Replace with your default model name
        model_name = data.get('model_name', 'efficientvit_m2')

        # Decode the image
        image = Image.open(
            BytesIO(base64.b64decode(image_data))).convert('RGB')

        # Preprocess the image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_image = preprocess_image(image).to(device)

        # Load the model
        num_classes = 4
        model = load_model(model_name, num_classes, device)

        classes = ['Cyst', 'Tumor', 'Stone', 'Normal']

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Return the prediction result
        return jsonify({"prediction": classes[prediction]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
@cross_origin(origins=allowed_origins)
def index():
    return "<h1>Kidney Disease Detector Backend</h1>"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
