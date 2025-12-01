
# Face Recognition Payment System

This project implements a complete face recognition pipeline using Python, OpenCV, and TensorFlow/Keras. It is designed for applications such as canteen payment systems, where real-time face recognition and user information display are required.

## Features
- **Face Acquisition:** Capture face images from a webcam and store them for each user.
- **Face Detection:** Detect and crop faces from images using Haar cascade classifiers.
- **Data Preprocessing:** Prepare and encode face images and labels for neural network training.
- **Model Building:** Construct a Convolutional Neural Network (CNN) for face classification.
- **Training & Evaluation:** Train the CNN, evaluate its accuracy, and save the model and weights.
- **Real-Time Recognition:** Perform live face recognition and display user info (ID, balance, consumption) on video frames.

## Folder Structure
```
face_recognition.ipynb         # Main Jupyter notebook with all steps
faceReco.weights.h5            # Saved model weights
final_faceReco.h5              # Saved complete model
haarcascade_frontalface_default.xml # Haar cascade for face detection
face_img/                      # Raw captured face images
small_face/                    # Cropped face images by user
README.md                      # Project documentation
```

## Setup Instructions
1. **Install Requirements:**
	- Python 3.x
	- OpenCV (`opencv-python`)
	- TensorFlow/Keras
	- scikit-learn
	- matplotlib
	- Pillow

2. **Prepare Haar Cascade:**
	- Ensure `haarcascade_frontalface_default.xml` is in the project directory.

3. **Run the Notebook:**
	- Open `face_recognition.ipynb` in Jupyter Notebook or VS Code.
	- Follow the notebook cells step by step:
	  1. **Face Acquisition:** Capture and save at least 100 face images per user.
	  2. **Face Detection:** Detect and crop faces, saving them in `small_face/<user_id>/`.
	  3. **Data Preprocessing:** Encode images and labels, split into train/test sets.
	  4. **Model Building:** Build and compile the CNN.
	  5. **Training:** Train the model and save it.
	  6. **Real-Time Recognition:** Run live recognition and display user info.

## Usage Notes
- For real-time recognition, you may need a font file (e.g., `simhei.ttf`) for displaying Chinese text.
- User information (ID, balance, consumption) should be provided in a dictionary for display.
- Images should be captured with varied angles and expressions for best results.

## Example User Info Mapping
```
Y_dic = {
	 'id': ['o1143', '1002'],
	 'balance': ['￥100', '￥80'],
	 'consumption': ['￥20', '￥40']
}
```

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

---

**Author:** Your Name

**Date:** December 2025
