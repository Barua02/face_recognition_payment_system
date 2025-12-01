"""
realtime_recognition.py
Real-time face recognition and user info display using webcam.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

def user_Info_get(y_pre, Y_dic):
    idx = np.argmax(y_pre)
    user_id = Y_dic['id'][idx]
    balance = Y_dic['balance'][idx]
    consumption = Y_dic['consumption'][idx]
    return user_id, balance, consumption

def draw_chinese(img, text, pos, color=(0,255,0), font_size=24):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def realtime_face_recognition(model_path='./final_faceReco.h5', cascade_path='haarcascade_frontalface_default.xml', Y_dic=None):
    if Y_dic is None:
        Y_dic = {
            'id': ['o1143', '1002'],
            'balance': ['￥100', '￥80'],
            'consumption': ['￥20', '￥40']
        }
    face_cascade = cv2.CascadeClassifier(cascade_path)
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (160, 160))
            face_input = np.expand_dims(face_resized, axis=0) / 255.0
            y_pre = model.predict(face_input)
            user_id, balance, consumption = user_Info_get(y_pre, Y_dic)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            info_text = f"工号/学号: {user_id}  余额: {balance}  消费: {consumption}"
            img = draw_chinese(img, info_text, (x, y-30), color=(255,0,0), font_size=24)
            break
        cv2.imshow('Real-Time Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_face_recognition()
