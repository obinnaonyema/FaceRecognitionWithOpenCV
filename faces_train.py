'''
In this file I train the model using images in the training folder.
If you have new images, add to the training folders under the right name 
(or add a folder for a new person), then rerun this training program.
'''
import cv2 as cv 
import numpy as np
import os

from numpy.core.numerictypes import obj2sctype

#people = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

DIR = r'Faces\train'

people = []
for name in os.listdir(DIR):
    people.append(name)
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            #detect face and append to features list
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done -----------------')
#print(f'Length of the features list = {len(features)}')
#print(f'Length of the labels list = {len(labels)}')

features = np.array(features, dtype = 'object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer on the featuers list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy',labels)
