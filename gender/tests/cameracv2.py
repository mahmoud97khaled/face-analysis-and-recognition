import cv2
from PIL import Image
import numpy as np
import torch
from gender_test import GenderClassifier  # Import your gender classifier

# Initialize the gender classifier (assuming the model is in 'models/gender_classifier.pth')
classifier = GenderClassifier(model_path='models/gender_classifier.pth')

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame from OpenCV BGR to PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Predict gender using the classifier
    classifier.predict(pil_image)

    # Convert the PIL image back to OpenCV format (numpy array)
    frame = np.array(pil_image)

    # Convert frame back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the camera feed with prediction
    cv2.imshow("Live Gender Prediction", frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()