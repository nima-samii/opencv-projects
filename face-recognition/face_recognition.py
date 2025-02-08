import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(r'face-recognition\data\haar_face.xml')

DIR = r'face-recognition\data\train'

people = []
for i in os.listdir(DIR):
    people.append(i)

fe_path = r'face-recognition\features.npy'
features = np.load(fe_path, allow_pickle=True)
la_path= r'face-recognition\labels.npy'
labels = np.load(la_path, allow_pickle= True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face-recognition\Face_trained.yml')

img_path = r'face-recognition\data\val\madonna\3.jpg'
img = cv.imread(img_path)
cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'person = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=1)

cv.imshow('Detected Face', img)

cv.waitKey(0)