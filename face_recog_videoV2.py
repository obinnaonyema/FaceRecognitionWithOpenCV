'''
This file implements face recognition using the face-recognition library which runs on the dlib
face recognition module.

From my experience it is an improvement on the performance of haarcascades.
'''
import cv2 as cv 
import numpy as np
import os
import face_recognition

faceLocations = np.load('faceLocations.npy', allow_pickle=True)
encodingList = np.load('encodingList.npy', allow_pickle=True)
peopleList = np.load('peopleList.npy', allow_pickle=True)

# make the encoded values float data type to avoid throwing errors
encodingList = encodingList.astype(float)

print(f'People list: {len(peopleList)}')

capture = cv.VideoCapture(0)

while True:
    _, img_array = capture.read()  
    img_rgb = cv.cvtColor(img_array,cv.COLOR_BGR2RGB)

    faceLoc0 = face_recognition.face_locations(img_rgb)

    # Check if faces were found
    if faceLoc0:
        faceLoc = faceLoc0[0]
        faceEncode = face_recognition.face_encodings(img_rgb,[faceLoc])[0]
        matches = face_recognition.compare_faces(encodingList, faceEncode)
        # print(f'Matches array size: {len(matches)}')
        faceDis = face_recognition.face_distance(encodingList, faceEncode)
        # print(f'Face distance array size: {len(faceDis)}')
        # Get index of best match (smallest distanct)
        matchIndex = np.argmin(faceDis)
        # print(f'Match index: {matchIndex}')

        # Check if the closest match exists in the array of encoded faces
        if matches[matchIndex]:
            name = peopleList[matchIndex]
            #print(name)
            y1,x2,y2,x1 = faceLoc
            #y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img_array,(x1-50,y1-50),(x2+50,y2+50),(0,255,0),2)
            cv.rectangle(img_array,(x1-50,y2-35),(x2+50,y2+50),(0,255,0),cv.FILLED)
            cv.putText(img_array,f'{name}',(x1-40,y2),cv.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
            cv.putText(img_array,f'Confidence {matchIndex}',(x1-40,y2+30),cv.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),2)
   

        else:
            #print(f'No faces found for {img}')
            y1,x2,y2,x1 = faceLoc
            cv.rectangle(img_array,(x1,y1),(x2,y2),(0,255,0),2)
            cv.putText(img_array,'Did not recognize the face',(x1,y1-100),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    
    else:
        #print(f'No faces found for {img}')
        cv.putText(img_array,'Did not recognize any faces',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    
    cv.imshow('Image',img_array)


    k = cv.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
cv.destroyAllWindows()