from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)
model = load_model("cnn_model.h5")

def classify_grade_by_aspect_ratio(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w == 0 or h == 0:
        return "Unknown"
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio >= 3.0:
        return "Slender"
    elif aspect_ratio >= 2.1:
        return "Medium"
    elif aspect_ratio >= 1.1:
        return "Bold"
    else:
        return "Round"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = "uploaded_image.jpg"
    file.save(filepath)
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    rice_types = ['Basmati', 'Arborio', 'Jasmine', 'Other']  # Adjust based on your model classes
    predicted_type = rice_types[class_idx]
    grade = classify_grade_by_aspect_ratio(filepath)
    return jsonify({'type': predicted_type, 'quality': grade})

if __name__ == '__main__':
    app.run(debug=True)
