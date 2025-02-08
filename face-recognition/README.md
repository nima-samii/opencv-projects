# Face Recognition System

This repository contains two Python scripts for training a face recognition model and using it to recognize faces in images. The system uses OpenCV for image processing and face detection, and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition.

## Scripts

1. **`faces_train.py`:** This script trains the face recognition model. It reads images from a training directory, detects faces using a Haar cascade classifier, extracts features from the faces, and trains an LBPH face recognizer. The trained model, features, and labels are saved to files.

2. **`face_recognition.py`:** This script uses the trained face recognition model to identify faces in new images. It loads the trained model, features, and labels, detects faces in an input image, and predicts the person's label using the LBPH recognizer. It then displays the image with the recognized person's name and a bounding box around the face.

## Requirements

- Python 3
- OpenCV
- NumPy 