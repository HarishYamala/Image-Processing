import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Image dimensions
img_height, img_width = 128, 128  # You can change this depending on image size
batch_size = 32

# Set dataset directory
data_dir = r'C:\Users\DELL\Desktop\major_ps2\Rice_Image_Dataset'

# Use ImageDataGenerator for loading and augmenting images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model = load_model("rice_classifier_cnn.h5")

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

    # Find the largest contour (most likely the grain)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    if w == 0 or h == 0:
        return "Unknown"

    aspect_ratio = max(w, h) / min(w, h)

    # Debug (optional): print dimensions and ratio
    # print(f"W: {w}, H: {h}, Aspect Ratio: {aspect_ratio:.2f}")

    # Classification logic
    if aspect_ratio >= 3.0:
        return "Slender"
    elif aspect_ratio >= 2.1:
        return "Medium"
    elif aspect_ratio >= 1.1:
        return "Bold"
    else:
        return "Round"

# Evaluate on validation set
loss, acc = model.evaluate(val_generator)
print("Validation Accuracy: {:.2f}%".format(acc * 100))

# Predict on a new image
from tensorflow.keras.preprocessing import image
def predict_image_with_grade(img_path):
    # Predict rice type using CNN
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = list(train_generator.class_indices.keys())
    predicted_type = class_labels[class_idx]

    # Predict rice grade using image processing
    predicted_grade = classify_grade_by_aspect_ratio(img_path)

    print(f"Predicted Type: {predicted_type}")
    print(f"Predicted Grade: {predicted_grade}")

predict_image_with_grade(r'C:\Users\DELL\Desktop\major_ps2\Rice_Image_Dataset\Basmati\Basmati (1001).jpg')

model.save('cnn_model.h5')