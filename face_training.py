import cv2
import numpy as np
from PIL import Image
import os


def run():
    path = "Dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        face_ids = []
        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert("L")
            img_numpy = np.array(PIL_img, "uint8")
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for x, y, w, h in faces:
                face_samples.append(img_numpy[y: y+h, x:x+w])
                face_ids.append(face_id)
        return face_samples, face_ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write("Trainer/trainer.yml")

    print(f"\n [INFO] {format(len(np.unique(ids)))} faces trained. Exiting Program")

    return get_images_and_labels(path)
