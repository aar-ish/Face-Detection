import cv2
import numpy as np
import face_recognition
import os

path = 'TestImages'
imageList = []
classNames = []
mylist = os.listdir(path)
#print(mylist)
for clas in mylist:
    curImg = cv2.imread(f'{path}/{clas}') # clas is the first img
    imageList.append(curImg)
    classNames.append(os.path.splitext(clas)[0])
print(classNames)

def Encoder(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnownFaces = Encoder(imageList)
print("Encoding Complete"+"\n"+str(len(encodeListKnownFaces))+" Images")

capture = cv2.VideoCapture(0)

while True: # to collect img one by one
    success,img = capture.read() # reduce size to speed process
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25) # 1/4th of actual size
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
# to find loc
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall,facesCurFrame)
# We may find multiple faces so will first locate face and send to the encoding function
    # To loop above thing together
    for encodeFace,faceLocation in zip(encodesCurFrame,facesCurFrame):
# 1 by 1 grab one face loc from faceloc
# zip is used to use them in same loop
        matches = face_recognition.compare_faces(encodeListKnownFaces,encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnownFaces,encodeFace)
        print(faceDistance)
# To get the best match we check the lowest reading
        matchesIndex = np.argmin(faceDistance)
# As index matches we show the accurate detected face
        if matches[matchesIndex]:
            name = classNames[matchesIndex].upper()
            print(name)
# Creating the dectection rectangle
            y1,x2,y2,x1 = faceLocation
# As previously we made img small to detect we need to make it normal again to display the green box
            y1, x2, y2, x1 = y1*4 , x2*4 , y2*4 , x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
# We display the detected img
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)