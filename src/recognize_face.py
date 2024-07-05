import cv2
import pickle
import sys
from utils import load_image, create_face_embedding, get_image_path
import numpy as np

def recognize_face(image_path1, image_path2):
    # Load the trained model
    with open('models/face_recognition_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    embedding1 = create_face_embedding(img1)
    embedding2 = create_face_embedding(img2)

    if embedding1 is None or embedding2 is None:
        return "Could not detect face in one or both images."

    combined_embedding = np.concatenate([embedding1, embedding2])
    prediction = clf.predict([combined_embedding])[0]
    probability = clf.predict_proba([combined_embedding]).max()

    if prediction == 1:
        result = "Same person"
    else:
        result = "Different persons"

    return f"{result} (Confidence: {probability:.2f})"

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python recognize_face.py <name1> <image_number1> <name2> <image_number2>")
        sys.exit(1)

    name1, img_num1, name2, img_num2 = sys.argv[1:]
    image_path1 = get_image_path(name1, img_num1)
    image_path2 = get_image_path(name2, img_num2)
    
    result = recognize_face(image_path1, image_path2)
    print(result)