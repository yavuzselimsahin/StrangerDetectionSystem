import cv2
import numpy as np
import os


def run():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer/trainer.yml")
    cascade_path = "Cascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_id = 0

    names = ["None", "Yavuz", "Ayca"]

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(int(min_w), int(min_h)),
        )
        for x, y, h, w in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (180, 105, 255), 2)
            face_id, accuracy = recognizer.predict(gray[y:y+h, x: x+w])

            if accuracy < 100:
                face_id = names[face_id]
                accuracy = f"{format(round(100 - accuracy))}"
            else:
                face_id = "Unknown"
                accuracy = f"{format(round(100 - accuracy))}"

            cv2.putText(
                img,
                str(face_id),
                (x+5, y-5),
                font,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                str(accuracy),
                (x+5, y+h-5),
                font,
                1,
                (255, 255, 0),
                1
            )
        cv2.imshow("Camera", img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
