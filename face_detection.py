"""
face_detection.py
Detect faces in images and save cropped faces for each user.
"""
import cv2
import os

def detect_and_crop_faces(large_face_dir='face_img', small_face_dir='small_face', cascade_path='haarcascade_frontalface_default.xml'):
    face_clf = cv2.CascadeClassifier(cascade_path)
    os.makedirs(small_face_dir, exist_ok=True)
    for img_name in os.listdir(large_face_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(large_face_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (160, 160))
            try:
                user_id = img_name.split('_')[1]
            except IndexError:
                user_id = 'unknown'
            user_folder = os.path.join(small_face_dir, user_id)
            os.makedirs(user_folder, exist_ok=True)
            save_path = os.path.join(user_folder, f"{os.path.splitext(img_name)[0]}_face{i+1}.jpg")
            cv2.imwrite(save_path, face_resized)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    detect_and_crop_faces()
