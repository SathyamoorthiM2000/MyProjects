import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

with_helmet_images = ["//content//with1.jpeg", "//content//with2.jpeg", "//content//with3.jpeg", "//content//with4.jpeg", "//content//with5.jpeg"]
without_helmet_images = ["//content//without1.jpeg", "//content//without2.jpeg", "//content//without 3.jpeg", "//content//without4.jpeg", "//content//without5.jpeg"]

def load_and_preprocess_image(image_path, resize_shape=(64, 64)):
    img = cv2.imread(image_path)  # Read the image
    img = cv2.resize(img, resize_shape)  # Resize to the desired shape for the model
    img = img.astype('float32') / 255.0  # Normalize the image
    return img

images = []
labels = []

for image_path in with_helmet_images:
    img = load_and_preprocess_image(image_path)
    images.append(img)
    labels.append(1)  

for image_path in without_helmet_images:
    img = load_and_preprocess_image(image_path)
    images.append(img)
    labels.append(0)  

X = np.array(images)
y = np.array(labels)

y = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test), verbose=0)

def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    label = np.argmax(prediction)  
    labels = ["Without Helmet", "With Helmet"]
    predicted_label = labels[label]
    print(f"Prediction: {predicted_label}")  
    
    if predicted_label == "Without Helmet":
        vehicle_number = input("Enter the vehicle number: ")
        fine_amount = 100  
        print(f"Sending message to the responsible person: 'Fine applied for vehicle {vehicle_number}. Fine amount: rs{fine_amount}'")

image_path = input("Enter the image path to predict: ")
predict_image(image_path)
        
        