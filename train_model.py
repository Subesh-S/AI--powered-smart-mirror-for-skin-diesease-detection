import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dataset folders
DATASET_PATH = "."

positive_folder = "pimple"
negative_folders = ["Acne", "hives", "melanoma", "no_disease"]

IMAGE_SIZE = 128

X = []
Y = []

def load_images_from_folder(folder, label):
    folder_path = os.path.join(DATASET_PATH, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0  # Normalization
        X.append(img)
        Y.append(label)

# Load positive images (pimple)
load_images_from_folder(positive_folder, 1)

# Load negatives (all others)
for folder in negative_folders:
    load_images_from_folder(folder, 0)

X = np.array(X)
Y = np.array(Y)

print("Total Images Loaded:", len(X))
print("Positive (pimple):", np.sum(Y == 1))
print("Negative:", np.sum(Y == 0))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("Training model...")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

model.save("pimple_binary_model.h5")
print("Model Saved Successfully: pimple_binary_model.h5")
