__author__ = 'zhengwang'

import numpy as np
import cv2
import socket
import os


class VideoStreamingTest(object):
    def __init__(self, host, port):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()

    def drawRectangleText(self, img, x, y, w, h, text, color=(0, 255, 0)):
        """Draw a rectangle with the given coordinates (rect) in the image"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        return img

    def performPrediction(self, face, recognizer, subjects):
        """Recognizes the face of a person in the image and
        returns information about that person"""

        # Recognize face
        # Note: predict() returns label=(int number, double confidence)
        prediction = recognizer.predict(face)

        # Search person who it's related to the number returned by predict()...
        if prediction[1] < 80:  # ...if confidence is small enough
            if prediction[0] in subjects:  # ... and if that number was registered in profiles.txt
                name = subjects[prediction[0]]
            else:
                name = "Not registered"
        else:
            name = "Unknown"  # otherwise, its an unknown person

        # Build text to be draw in the image (with confidence
        # value converted to percentage)
        confidence = 100 - prediction[1]
        recognition_info = name + " (" + format(confidence, ".2f") + "%)"

        return recognition_info

    def loadSubjects(self):
        relations = {}

        if not os.path.isfile("profiles.txt"):
            print("No se encontro archivo de perfiles")
            exit(0)
        file = open("profiles.txt", "r")
        for line in file:
            line = line.replace("\n", "")
            relations[int(line[0])] = line.replace(line[0] + "-", "")
        file.close()

        return relations

    def loadModel(self):
        if not os.path.isfile("model.yml"):
            print("No se encontro archivo de modelo")
            exit(0)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read("model.yml")

        return face_recognizer

    def streaming(self):
        # DEFAULT SIZES
        minFaceSize = 45  # 40 is good for PiCamera detection up to 4 meters
        maxFaceSize = 155  # up to 160 (smaller size, better performance)

        # LOAD RESOURCES
        # Load detectors
        frontal_detector = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_frontalface_default.xml')
        stop_sign_detector = cv2.CascadeClassifier('xml-files/haarcascades/stop_sign.xml')
        # frontal_detector = cv2.CascadeClassifier('xml-files/haarcascades/traffic_light.xml')
        lateral_detector = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_profileface.xml')

        # Load subjects (for prediction)
        subjects = self.loadSubjects()
        # Load trained model
        model = self.loadModel()

        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")

            # need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    frontal_faces = frontal_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=10,
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

                    lateral_faces = lateral_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=10,
                        minSize=(minFaceSize, minFaceSize),
                        maxSize=(maxFaceSize, maxFaceSize)
                    )

                    # Draw a rectangle around the faces
                    for (x, y, w, h) in frontal_faces:
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Crop face
                        cropped_face = gray[y:y + w, x:x + h]
                        # Perform recognition of face
                        recognition_info = self.performPrediction(cropped_face, model, subjects)
                        
                        print(recognition_info)
                        # Draw rectangle and text
                        frame = self.drawRectangleText(frame, x, y, h, w, recognition_info)

                    # for (x, y, w, h) in lateral_faces:
                    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    for (x, y, w, h) in stop_signs:
                        frame = self.drawRectangleText(frame, x, y, h, w, 'stop', (0, 0, 255))
                        print('stop sign detected')

                    # Debug face range rectangles
                    cv2.rectangle(frame, (0, 0), (0 + maxFaceSize, 0 + maxFaceSize), (255, 0, 0))  # Max size
                    cv2.rectangle(frame, (maxFaceSize, 0), (maxFaceSize + minFaceSize, 0 + minFaceSize), (0, 0, 255))  # Min size

                    # Display the resulting frame
                    cv2.imshow('video', frame)

                    # Press 'q' to stop face recognition
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
    # host, port
    import sys
    h, p = sys.argv[1].split(' ')[0], 8000
    print("server running on", sys.argv[1].split(' ')[0])
    VideoStreamingTest(h, p)
    
