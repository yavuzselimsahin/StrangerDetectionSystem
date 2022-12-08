import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(20, 20)
    )
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = image[y: y+h, x: x+w]

    cv2.imshow("Video", image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()
