import cv2 as cv

img = cv.imread(r'face-detection\data\lady.jpg')
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier(r'face-detection\data\haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= 4)

print(f'Number of face(s) found= {len(faces_rect)}')

for i,j,w,h in faces_rect:
    cv.rectangle(img, (i,j), (i+w,j+h), (0,255,0), thickness=1)

cv.imshow('Detected face(s)', img)

cv.waitKey(0)