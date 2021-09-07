'''
In this file I train the model using images in the training folder.
If you have new images, add to the training folders under the right name 
(or add a folder for a new person), then rerun this training program.
'''
import cv2 as cv 
import numpy as np
import os
import face_recognition

from numpy.core.numerictypes import obj2sctype

#people = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

DIR = r'Faces\train'

people = []
for name in os.listdir(DIR):
    people.append(name)

faceLocations = []
encodingList = []
peopleList = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)

        for img in os.listdir(path):
            print(f'Processing for: {img}')
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
            
            #detect face
            faces_locs0 = face_recognition.face_locations(img_rgb)

            if faces_locs0:
                faces_locs = faces_locs0[0]
                #append faces to lists
                faceLocations.append(faces_locs)
                print(f'Obtained face locations for: {img}. Number of faces: {len(faces_locs)//4}')
                faces_encode = face_recognition.face_encodings(img_rgb,[faces_locs])
                print(f'Encoded faces for {person}: {img}')
                encodingList.append(faces_encode[0])
                peopleList.append(person)
            else:
                print(f'No faces found for {person:} {img}')


create_train()
print('Training done -----------------')
print(f'Number of faces: {len(faceLocations)}')
print(f'Number of encodings: {len(encodingList)}')
print(f'Number of labels: {len(peopleList)}')

faceLocations = np.array(faceLocations, dtype = 'object')
encodingList = np.array(encodingList, dtype = 'object')
peopleList =np.array(peopleList)


np.save('faceLocations.npy', faceLocations)
np.save('encodingList.npy',encodingList)
np.save('peopleList.npy',peopleList)
print(f'\n Training files saved')
