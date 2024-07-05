import os
import numpy as np
import pickle
from utils import load_image, create_face_embedding, load_pairs, get_image_path
from sklearn.model_selection import train_test_split

def prepare_dataset(pair_file):
    pairs = load_pairs(pair_file)
    X = []
    y = []

    for pair in pairs:
        if len(pair) == 3:
            name, img1, img2 = pair
            path1 = get_image_path(name, img1)
            path2 = get_image_path(name, img2)
            label = 1  # Same person
        else:
            name1, img1, name2, img2 = pair
            path1 = get_image_path(name1, img1)
            path2 = get_image_path(name2, img2)
            label = 0  # Different persons

        image1 = load_image(path1)
        image2 = load_image(path2)
        
        embedding1 = create_face_embedding(image1)
        embedding2 = create_face_embedding(image2)
        
        if embedding1 is not None and embedding2 is not None:
            X.append(np.concatenate([embedding1, embedding2]))
            y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Prepare training data
    X_train, y_train = prepare_dataset('data/pairsDevTrain.txt')

    # Prepare testing data
    X_test, y_test = prepare_dataset('data/pairsDevTest.txt')

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Save the prepared dataset
    with open('data/prepared_dataset.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }, f)

    print("Dataset prepared and saved.")