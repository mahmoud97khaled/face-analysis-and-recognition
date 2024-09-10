import cv2
from PIL import Image
import numpy as np
import torch
from test_without_camera import EmotionDetector  


classifier = EmotionDetector(model_path='res34_res34.pth')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to grab frame")
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (the face) from the frame
        face = frame[y:y+h, x:x+w]

        # Convert face from OpenCV BGR to PIL RGB for emotion prediction
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)

        # Predict emotion using the classifier
        predicted_emotion, confidence = classifier.predict(pil_face)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the emotion and confidence on the frame
        emotion_text = f'{predicted_emotion} ({confidence:.2f})'
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the camera feed with prediction text and face detection
    cv2.imshow("Live Emotion Detector", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
