import cv2
import os

path=r'D:\暫存'
face_cascede=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
num=0
for filename in os.listdir():
    print(filename)
    if filename.endswith('.jpg'):
        img=cv2.imread(filename)
        faces=face_cascede.detectMultiScale(img)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)
        cv2.imwrite(str(num)+'.jpg', img)
        cv2.imshow('girl', img)
    num+=1

        