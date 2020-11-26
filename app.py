import numpy as np
import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eyescascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
eyescascadeglasses = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
smileIdent = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(-1)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  
    # Draw the rectangle around each face
  
    for (x,y,w,h) in faces:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        smiles = smileIdent.detectMultiScale(roi_gray)
        eyes = eyescascade.detectMultiScale(roi_gray)
        eyesg = eyescascadeglasses.detectMultiScale(roi_gray)
    
            



        for (ex,ey,ew,eh) in eyesg:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()

