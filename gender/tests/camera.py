import cv2
from PIL import Image
import numpy as np
from gender_test import GenderClassifier  # Import your gender classifier

# Load the pre-trained Haar Cascade face detection model
def get_images():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the gender classifier
    classifier = GenderClassifier(model_path='/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/models/gender_classifier.pth')

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    while True:
        images = []
        # Capture frame-by-frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        # # Detect faces in the frame with more focused parameters
        # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80))


        # If no face is detected, just show the frame
        if len(faces) == 0:
            cv2.imshow("Live Gender Prediction", frame)
        else:
            # Loop over all detected faces (usually one face in most cases)
            for (x, y, w, h) in faces:
                # Extract the face region from the frame
                face_frame = frame[y:y+h, x:x+w]

                # Resize the face region to a consistent size (e.g., 224x224)
                # face_resized = cv2.resize(face_frame, (128, 128))

                # Convert the face frame from OpenCV BGR to PIL RGB
                face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                images.append(face_rgb)

                prediction = classifier.predict(face_rgb)

                # # Add prediction text to the frame
                if 0.3< prediction < 0.7:
                    gender_text = f'Prediction : ------'

                elif prediction > 0.5:
                    gender_text = f'Prediction: Female ({prediction:.2f})'
                else:
                    gender_text = f'Prediction: Male ({1 - prediction:.2f})'

                # Draw a rectangle around the face and add the text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, gender_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the frame with face detection and prediction
            cv2.imshow("Live Gender Prediction", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
