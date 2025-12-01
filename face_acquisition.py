"""
face_acquisition.py
Capture and save face images from webcam for registration.
"""
import cv2
import os

def capture_faces(face_img_dir='face_img', max_images=100):
    os.makedirs(face_img_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        cap.release()
        return
    user_id = input("Enter user ID (e.g., student ID): ")
    count = 0
    print("Press 's' to save, 'q' to quit.")
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Capture Face', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_name = f"user_{user_id}_{count+1}.jpg"
            cv2.imwrite(os.path.join(face_img_dir, img_name), gray)
            print(f"Saved: {img_name}")
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
