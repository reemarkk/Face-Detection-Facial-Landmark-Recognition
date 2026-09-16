
# Face Detection & Facial Landmark Recognition

## Description

This project is a Python-based computer vision application that detects human faces, eyes, and facial landmarks in images.

It uses:

- **OpenCV** for image processing and eye detection
- **Dlib** for face detection and facial landmark prediction
- **NumPy** for numerical image processing
- **Pillow** for image loading and conversion
- **Matplotlib** for displaying the processed images

## Problem

The goal of this project is to automatically identify human faces in images and determine important points on each detected face.

For every detected face, the program identifies **68 facial landmarks**, including points around:

- Eyes
- Eyebrows
- Nose
- Mouth
- Jawline

The program also detects the eyes and displays the detected faces and landmarks on the original images.

This type of facial landmark detection can be used as a basic component of applications such as facial recognition, face tracking, emotion analysis, augmented reality, and other computer vision systems.

## Required Model

This project requires the Dlib **68-point facial landmark predictor model**:

`shape_predictor_68_face_landmarks.dat`

This file is a pre-trained machine learning model that Dlib uses to locate the 68 facial landmark points on a detected face.

The model file is **not included in this repository** because of its large size.

### Download the model

Download the following file:

`shape_predictor_68_face_landmarks.dat.bz2`

from the official Dlib website.

After downloading it, extract the file:

```bash
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

## Facial Landmarks

The 68 facial landmarks detected by the Dlib facial landmark predictor are shown below:

<img width="602" height="672" alt="68 facial landmarks" src="https://github.com/user-attachments/assets/1851edf6-2c1c-479e-bb4d-0699f747fdcd" />

## Results

The final results of face detection, eye detection, and facial landmark detection are shown below:

<img width="830" height="412" alt="Face detection and facial landmarks results" src="https://github.com/user-attachments/assets/b9795f58-22f9-40d3-a3ea-05fb9d500b75" />


