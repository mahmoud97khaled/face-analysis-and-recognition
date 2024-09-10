from tests.emotion_test_without_camera import EmotionDetector 
from tests.gender_test import GenderClassifier

import cv2
import numpy as np

# Load the pre-trained Haar Cascade face detection model
def get_images():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the gender classifier
    gender_classifier = GenderClassifier(model_path='/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/models/gender_classifier.pth')
    emotion_classifier = EmotionDetector("/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/emotion recognition/models/res34_res34.pth")

    # Initialize the camera
    camera = cv2.VideoCapture(0)
    counter = 0
    current_str = ''
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if counter % 10 == 0:
            counter += 1
            

            if not ret:
                print("Failed to grab frame")
                break

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # If no face is detected, just show the frame
            if len(faces) == 0:
                cv2.imshow("Live Gender and Emotion Prediction", frame)
            else:
                # Loop over all detected faces
                for (x, y, w, h) in faces:
                    # Extract the face region from the frame
                    face_frame = frame[y:y+h, x:x+w]

                    # Convert the face frame from OpenCV BGR to RGB
                    face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

                    # Predict gender using the gender classifier
                    gprediction = gender_classifier.predict(face_rgb)
                    # Predict emotion using the emotion classifier
                    eprediction, confidence = emotion_classifier.predict(face_rgb)

                    # Process the gender prediction
                    if 0.3 < gprediction < 0.7:
                        gender_text = 'Prediction: Uncertain'
                    elif gprediction > 0.5:
                        gender_text = f'Prediction: Female ({gprediction:.2f})'
                    else:
                        gender_text = f'Prediction: Male ({1 - gprediction:.2f})'

                    # Process the emotion prediction
                    emotion_text = f'Emotion: {eprediction} ({confidence:.2f})'

                    # Combine both gender and emotion texts
                    combined_text = f'{gender_text}, {emotion_text}'
                    current_str = combined_text

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Add both the gender and emotion predictions as text above the face
                    cv2.putText(frame, combined_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display the frame with predictions
                cv2.imshow("Live Gender and Emotion Prediction", frame)
        else:
            counter += 1
            cv2.imshow("Live Gender and Emotion Prediction", frame)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add both the gender and emotion predictions as text above the face
            cv2.putText(frame, combined_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Live Gender and Emotion Prediction", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
get_images()
