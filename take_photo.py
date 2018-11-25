
import time
import cv2
import os

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

    if not os.path.isfile("model/profiles.txt"):
        print("No se encontro archivo de perfiles")
        exit(0)
    file = open("model/profiles.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        relations[int(line[0])] = line.replace(line[0] + "-", "")
    file.close()

    return relations


def loadModel():
    if not os.path.isfile("model/model.yml"):
        print("No se encontro archivo de modelo")
        exit(0)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("model/model.yml")

    return face_recognizer


minFaceSize = 50  # (50-150) is good for PiCamera detection up to 4 meters
maxFaceSize = 250

# allow the camera to warmup
time.sleep(0.1)

# grab an image from the camera
image = cv2.imread('/home/pi/Desktop/FaceRecon/photos/test.jpg',0)
# LOADING RESOURCES
# Relations number-person (smth like {1: "Fernando", 2: "Esteban", ...})
subjects = loadSubjects()
# Trained model
model = loadModel()
# Load detectors
faces_detector = cv2.CascadeClassifier(
    'xml-files/lbpcascades/lbpcascade_frontalface.xml')
stop_sign_detector = cv2.CascadeClassifier(
    'xml-files/haarcascades/stop_sign.xml')
# frontal_detector = cv2.CascadeClassifier('xml-files/haarcascades/traffic_light.xml')
lateral_detector = cv2.CascadeClassifier(
    'xml-files/haarcascades/haarcascade_profileface.xml')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting faces in image
faces = faces_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(minFaceSize, minFaceSize),
    maxSize=(maxFaceSize, maxFaceSize)
    )

stop_signs = stop_sign_detector.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(minFaceSize, minFaceSize),
    maxSize=(maxFaceSize, maxFaceSize)
    )

for (x, y, w, h) in stop_signs:
    image = drawRectangleText(image, x, y, h, w, 'stop sign', (0, 0, 255))
    print('stop sign')
    cv2.imwrite('/home/pi/Desktop/FaceRecon/photos/stop_sign.jpg', image)
    os.system('sudo python3 /home/pi/Desktop/RPiControllers/send_email.py ' +
               '/home/pi/Desktop/FaceRecon/photos/stop_sign.jpg')

    # PROCESSING EACH FACE IN image
    for (x, y, h, w) in faces:
        # Crop face
        cropped_face = gray[y:y + w, x:x + h]
        # Perform recognition of cropped face
        recognition_info = performPrediction(cropped_face, model, subjects)
        # Draw rectangle and text
        print(recognition_info)
        image = drawRectangleText(image, x, y, h, w, recognition_info)
        cv2.imwrite('/home/pi/Desktop/FaceRecon/photos/' +
                    recognition_info.split(' ')[0] + '.jpg', image)
        os.system('sudo python3 /home/pi/Desktop/RPiControllers/send_email.py ' +
                  '/home/pi/Desktop/photos/' + recognition_info.split(' ')[0] + '.jpg')


# display the image on screen and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)
