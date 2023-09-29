#this is a face detector code
#press 'x' to exit the application
#This is made by Lee Henriques

import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#img = cv2.imread("IMG-20171104-WA0011.jpg")
video = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = video.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256) ,randrange(256)), 5)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),5)

    cv2.imshow("Lee's Face detector app", frame)
    key = cv2.waitKey(1)

    if key==120 or key==88:  #press x to close application
        break

video.release()
cv2.destroyAllWindows()



print("code completed")