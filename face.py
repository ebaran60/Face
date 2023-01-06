import os.path

import numpy as np
import cv2
import sys



faceCascade = cv2.CascadeClassifier(os.path.join('haarcascade_frontalface_default.xml'))

# Read the image
image = cv2.imread("oldphoto.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print ("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 6)

cv2.imshow("Faces found" ,image)
cv2.waitKey(0)
