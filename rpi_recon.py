import cv2
import numpy as np
import os
import shutil
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def drawRectangleText(img, x, y, w, h, text, color=(0, 255, 0)):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
    return img

def performPrediction(face, recognizer, subjects):
    """Recognizes the face of a person in the image and
    returns information about that person"""

    # Recognize face
    # Note: predict() returns label=(int number, double confidence)
    prediction = recognizer.predict(face)

    # Search person who it's related to the number returned by predict()...
    if prediction[1] < 100:  # ...if confidence is small enough
        if prediction[0] in subjects:  # ... and if that number is registered in profiles.txt
            name = subjects[prediction[0]]
        else:
            name = "Not registered"
    else:
        name = "Unknown"  # otherwise, its an unknown person

    # Build text to be draw in the image (with confidence
    # value converted to percentage)
    confidence = 100 - prediction[1]
    recognition_info = name + " - " + format(confidence, ".2f") + "%"

    return recognition_info


def loadSubjects():
    relations = {}

    if not os.path.isfile("/home/pi/Desktop/FaceRecon/model/profiles.txt"):
        print("No se encontro archivo de perfiles")
        exit(0)
    file = open("/home/pi/Desktop/FaceRecon/model/profiles.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        relations[int(line[0])] = line.replace(line[0] + "-", "")
    file.close()

    return relations


def loadModel():
    if not os.path.isfile("/home/pi/Desktop/FaceRecon/model/model.yml"):
        print("No se encontro archivo de modelo")
        exit(0)

    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.read("/home/pi/Desktop/FaceRecon/model/model.yml")

    return face_recognizer


def startRecon():
    # DEFINING PARAMETERS (for best performance)
    minFaceSize = 50  # (50-150) is good for PiCamera detection up to 4 meters
    maxFaceSize = 250

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # LOADING RESOURCES
    # Relations number-person (smth like {1: "Fernando", 2: "Esteban", ...})
    subjects = loadSubjects()
    # Trained model
    model = loadModel()

    # Load detectors
    faces_detector = cv2.CascadeClassifier('/home/pi/Desktop/FaceRecon/xml-files/lbpcascades/lbpcascade_frontalface.xml')
    stop_sign_detector = cv2.CascadeClassifier('/home/pi/Desktop/FaceRecon/xml-files/haarcascades/stop_sign.xml')
    # frontal_detector = cv2.CascadeClassifier('xml-files/haarcascades/traffic_light.xml')
    lateral_detector = cv2.CascadeClassifier('/home/pi/Desktop/FaceRecon/xml-files/haarcascades/haarcascade_profileface.xml')

    # Video stream (here we can capture an RPi stream instance)
    for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = image.array
        # Convert frame to gray scale for better detection accuracy
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in frame
        faces = faces_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(minFaceSize, minFaceSize),
            maxSize=(maxFaceSize, maxFaceSize)
        )

        stop_signs = stop_sign_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(minFaceSize, minFaceSize),
            maxSize=(maxFaceSize, maxFaceSize)
        )

        for (x, y, w, h) in stop_signs:
            frame = drawRectangleText(frame, x, y, h, w, 'stop sign', (0, 0, 255))
            print('stop sign')
            cv2.imwrite('/home/pi/Desktop/FaceRecon/photos/stop_sign.jpg', frame)
            os.system('sudo python3 /home/pi/Desktop/RPiControllers/camera/send_email.py ' + '/home/pi/Desktop/FaceRecon/photos/stop_sign.jpg')
        
        # PROCESSING EACH FACE IN FRAME
        for (x, y, h, w) in faces:
            # Crop face
            cropped_face = gray_frame[y:y + w, x:x + h]
            # Perform recognition of cropped face
            recognition_info = performPrediction(cropped_face, model, subjects)
            # Draw rectangle and text
            print(recognition_info)
            frame = drawRectangleText(frame, x, y, h, w, recognition_info)
            cv2.imwrite('/home/pi/Desktop/FaceRecon/photos/' + recognition_info.split(' ')[0] + '.jpg', frame)
            os.system('sudo python3 /home/pi/Desktop/RPiControllers/camera/send_email.py ' + '/home/pi/Desktop/FaceRecon/photos/' + recognition_info.split(' ')[0] + '.jpg')
        
        # Draw rectangles indicating smallest and biggest space that can be detected as a face
        # cv2.rectangle(frame, (0, 0), (0 + minFaceSize, 0 + minFaceSize), (0, 0, 255))  # Min size
        # cv2.rectangle(frame, (0, 0), (0 + maxFaceSize, 0 + maxFaceSize), (255, 0, 0))  # Max size

        # Display resulting frame
        cv2.imshow('Video feed', frame)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # Recognition will stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release resources (webcam or RPi stream)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    startRecon()