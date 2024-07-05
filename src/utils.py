import cv2
import numpy as np
import dlib
import os
import zipfile
from io import BytesIO

# Initialize dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

zip_path = "data/lfw-funneled.zip"

def load_image(image_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        full_path = os.path.join("lfw-funneled", image_path)
        with zip_ref.open(full_path) as file:
            img_array = np.frombuffer(file.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_face_embedding(image):
    dets = face_detector(image, 1)
    if len(dets) > 0:
        shape = shape_predictor(image, dets[0])
        face_embedding = np.array(face_recognition_model.compute_face_descriptor(image, shape))
        return face_embedding
    return None

def load_pairs(pair_file):
    pairs = []
    with open(pair_file, 'r') as f:
        for line in f:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs[1:]  # Skip the header

def get_image_path(name, image_number):
    return f"{name}/{name}_{int(image_number):04d}.jpg"