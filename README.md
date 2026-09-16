
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
