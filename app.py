'''
Challenges.
I need to get the video stream from the client browser. Using cv.VideoCapture(0)
opens the webcam on the server.

Browsers have a getUserMedia() function that can be used to obtain video stream.
It appears I'll have to use the OpenCV.js instead of the python version to do my model training smoothly.
'''
import os
from flask import Flask, request, redirect, url_for, render_template, flash, Response
import requests
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import base64
from datetime import datetime
import logging

ALLOWED_EXTENSIONS = {'jpg', 'png','.jpeg'}
UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = 'recognizer'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024 #6mb max file size


logging.basicConfig(filename='logs/log{}.log'.format(datetime.now().strftime("%m_%d_%Y")), level=logging.DEBUG)



@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        print('Received call after image processing')
        print(request.get_json())
        return render_template('index.html',data='hi')
    return render_template('index.html')  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET','POST'])
def upload():   
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File extension not supported')
            return redirect(request.url)            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            print('Received uploaded image {} for processing'.format(filename))
            process_image = predict_faces(Photo = os.path.join(UPLOAD_FOLDER, filename))
            print('Processed file name after processing uploaded file is {}'.format(str(process_image)))
            return render_template('upload.html',data=process_image)
    return render_template('upload.html') 

@app.route('/snapshot', methods=['GET','POST'])
def snapshot():
    if request.method =='POST':
        # take out the encoding information at the beginning
        image_received = request.get_json()['image'].split(',')[1]
        image_bytes = bytes(image_received,'utf-8')
        image_decoded = base64.decodebytes(image_bytes)
        filename=r'static\uploads\snapshot.jpg'
        image_file = open(filename,'wb')
        image_file.write(image_decoded)
        # call the video endpoint to process the image
        process_image = requests.post('http://localhost:5000/video', json={'filename':filename})
        if Response(process_image).status=='200 OK':
            print('Response from image recognition call for webcam image: ', Response(process_image).status)
            data = process_image.content.decode('UTF-8')
            print('Processed file name is {}'.format(str(data)))
        #return Response(requests.get('http://localhost:5000/result', params={'description':data}),status=200,mimetype='application/json') 
        #render_template('result.html', data=data) 
        return os.path.join('static/downloads',data)  
    else:
        print('Receiving request from ', request.remote_addr)
        return render_template('snapshot.html')

@app.route('/video', methods=['POST'])
def video():
    #return Response(predict_faces(), mimetype='multipart/x-mixed-replace; boundary=img')
    Photo = request.get_json()['filename']
    print('uploaded file name is: ',Photo)
    process_name = predict_faces(Photo)
    print(process_name)
    return Response(response=process_name,status=200,mimetype='application/json')

def predict_faces(Photo=None):
    DIR = r'Faces\train'
    people = []
    for name in os.listdir(DIR):
        people.append(name)
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    #commenting these guys out because of the allow_pickle=False error
    #features = np.load('features.npy')
    #labels = np.load('labels.npy')

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')
    if Photo==None:
        Photo = r'static\uploads\snapshot.jpg'
    img = cv.imread(Photo)
    #cap = cv.VideoCapture(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #cv.imshow('Person', gray)

        # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=6)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label])+' '+str(round(confidence)) if confidence<70 else 'I don\'t know you', (x-50,y-50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
        cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0), thickness=2)
            
    if len(faces_rect)==0:
        cv.putText(img, 'I didn\'t find faces. My eye is paining me', (20,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        print('Couldn\'t detect faces')
    # while True:
    #     _, img = cap.read()
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     cv.imshow('Person', gray)

    #     # Detect the face in the image
    #     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=4)

    #     for (x,y,w,h) in faces_rect:
    #         faces_roi = gray[y:y+h,x:x+w]

    #         label, confidence = face_recognizer.predict(faces_roi)
    #         print(f'Label = {people[label]} with a confidence of {confidence}')

    #         cv.putText(img, str(people[label])+' '+str(round(confidence)) if confidence<60 else 'I don\'t know you', (x-50,y-50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    #         cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0), thickness=2)

    #     #cv.imshow('Detected Face', img)
    #     #k = cv.waitKey(30) & 0xff
    #     #if k==27:
    #         #break
    #     ret, buffer = cv.imencode('.jpg', img)
    #     img = buffer.tobytes()
    #     yield(b'--img\r\n'
    #             b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        
    # cap.release()
    # cv.destroyAllWindows()
    cv.imshow('Detected Face', img)
    
    processed_file='processed{}.jpg'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    print(os.path.join('static\downloads',processed_file))
    cv.imwrite(os.path.join('static\downloads',processed_file),img)
    return processed_file
    
    
    #yield(b'--img\r\n'
    #        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    #return Response(image,mimetype='multipart/x-mixed-replace; boundary=img')
    



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
