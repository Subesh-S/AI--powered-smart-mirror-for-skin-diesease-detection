
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder




# Define image size and dataset paths
image_size = (100, 100)
pimple_folder = r'D:\mirror detection files\mirror_detection\pimple'
melanoma_folder = r'D:\mirror detection files\mirror_detection\melanoma'
no_disease_folder = r'D:\mirror detection files\mirror_detection\no_disease'  # Changed the folder name

# Load images from the datasets
pimple_images = []
melanoma_images = []
no_disease_images = []  # Changed the variable name
labels = []

# Load pimple images
for filename in os.listdir(pimple_folder):
    img = cv2.imread(os.path.join(pimple_folder, filename))
    img = cv2.resize(img, image_size)
    pimple_images.append(img)
    labels.append('pimple')

# Load melanoma images
for filename in os.listdir(melanoma_folder):
    img = cv2.imread(os.path.join(melanoma_folder, filename))
    img = cv2.resize(img, image_size)
    melanoma_images.append(img)
    labels.append('melanoma')

# Load no-disease images
for filename in os.listdir(no_disease_folder):  # Changed the folder name
    img = cv2.imread(os.path.join(no_disease_folder, filename))
    img = cv2.resize(img, image_size)
    no_disease_images.append(img)  # Changed the variable name
    labels.append('no_disease')  # Changed the label

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Combine the datasets
combined_images = pimple_images + melanoma_images + no_disease_images

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_images, labels, test_size=0.2, random_state=42)

# Extract features (for simplicity, we'll use flattened pixel values)
X_train = [img.flatten() for img in X_train]
X_test = [img.flatten() for img in X_test]

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Now, you can use the trained model to classify images into 'pimple,' 'melanoma,' and 'no_disease' categories.


import cv2
import numpy as np
import tkinter as tk
from tkinter import Label

# Function to open a website link
def open_website_link(link):
    import webbrowser
    webbrowser.open(link)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify another camera index

# Set camera properties
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load the face detection classifier (Haar Cascade or another face detection method)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a Tkinter window for the clickable link
root = tk.Tk()
root.geometry("800x600")  # Set the window size

# Create a Tkinter label for the link and center it
link_label = Label(root, text="", fg="blue", cursor="hand2")
link_label.pack(expand=True, fill='both')  # Center the label both horizontally and vertically

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region for classification
        face = frame[y:y+h, x:x+w]
        img = cv2.resize(face, (100, 100))  # Adjust the size as needed

        # Preprocess the image for classification
        img = img.flatten().reshape(1, -1)

        # Make a prediction using your trained model (clf)
        prediction = clf.predict(img)

        # Determine the label and website link based on the prediction
        if prediction == 0:
            label = 'Pimple'
            website_link = ''
        elif prediction == 1:
            label = 'melanoma'
            website_link = ''
        else:
            label = 'look healthy'
            website_link = ''

        # Update the link label's text
        link_label.config(text=f'Tips: {label}')
        link_label.bind("<Button-1>", lambda e, link=website_link: open_website_link(link))

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the classification result on the frame
        cv2.putText(frame, f'Result: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Result', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Start the Tkinter main loop
root.mainloop()