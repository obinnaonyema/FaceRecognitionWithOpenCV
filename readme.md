# Face Recognition With OpenCV

In this project we implement face recognition with OpenCV. The first version uses haarcascades as the model to recognize faces. The second version uses the face-recognition module.

## Requirements
### Version 1
* OpenCV
* Numpy
* Webcam.js (for the front end)
* Flask
* werkzeug.utils
* base64

### Version 2
* OpenCV
* Numpy
* Face-recognition

## Implementation & Results
### Version 1
The first version was implemented using [haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades).

A front end was developed using Flask to accept input via a webcam, process the image and return a processed image with recognition result.

![Webcam page](static/assets/webcam_page.JPG)

I had trouble implementing the webcam data capture initially using OpenCV for python. I discovered there was the JS version available as OpenCV.js. However I implemented image capture with Webcam.js and implemented scripts using [this tutorial](https://makitweb.com/how-to-capture-picture-from-webcam-with-webcam-js/).

The captured image is sent to the Flask app for processing via Ajax `XMLHttpRequest()`. The `POST` response contains the processed image filename which the `snapshots.html` page calls to load and display the processed image.

Save the features and labels arrays as [.npy files](https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/) so that they can be reused without having to train the model every time a face recognition task is to be done.