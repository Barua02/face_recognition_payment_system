"""
data_preprocessing.py
Load cropped face images, encode labels, and split data for training/testing.
"""
import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_and_preprocess(small_face_dir='small_face'):
    X = []
    Y = []
    for user_id in os.listdir(small_face_dir):
        user_folder = os.path.join(small_face_dir, user_id)
        if not os.path.isdir(user_folder):
            continue
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            Y.append(user_id)
    X = np.array(X)
    Y = np.array(Y)
    label_encoder = preprocessing.LabelEncoder()
    Y_num = label_encoder.fit_transform(Y)
    Y_cat = to_categorical(Y_num)
    x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42, stratify=Y_num)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    return x_train, x_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    load_and_preprocess()
