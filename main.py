import cv2
import numpy as np
import dlib
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat'
)

def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    landmarks = []

    for face in faces:
        shape = predictor(gray, face)

        landmarks.append([
            (p.x, p.y) for p in shape.parts()
        ])

    return faces, landmarks

def plot_landmarks(image, faces, landmarks):
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    for face in faces:
        x, y, w, h = (
            face.left(),
            face.top(),
            face.width(),
            face.height()
        )

        plt.gca().add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                color='blue',
                linewidth=2
            )
        )

    for face_landmarks in landmarks:
        for (x, y) in face_landmarks:
            plt.plot(
                x,
                y,
                'ro',
                markersize=2
            )

    plt.axis('off')
    plt.show()

def detect_eyes(image, faces):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    eyes = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        detected_eyes = eye_cascade.detectMultiScale(
            roi_gray
        )

        for (ex, ey, ew, eh) in detected_eyes:
            eyes.append(
                (x + ex, y + ey, ew, eh)
            )

    return eyes

def process_images(image_urls):
    for url in image_urls:
        image = download_image(url)

        faces, landmarks = detect_landmarks(image)

        eyes = detect_eyes(
            image,
            [
                (
                    f.left(),
                    f.top(),
                    f.width(),
                    f.height()
                )
                for f in faces
            ]
        )

        plot_landmarks(
            image,
            faces,
            landmarks
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                image,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

        plt.imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        plt.axis('off')
        plt.show()

image_urls = [
    "https://img.freepik.com/premium-photo/close-up-portrait-serious-gloomy-bearded-man-with-stylish-hairstyle-posing_160360-855.jpg?w=996",
    "https://img.freepik.com/free-photo/close-up-handsome-man-portrait_23-2148677648.jpg?t=st=1733903758~exp=1733907358~hmac=fa398dfdcb80b11d7f615852d4a105eecbc4124594ff9c92b34f9fb78ca58453&w=996",
    "https://img.freepik.com/free-photo/close-up-blond-man-posing-serious_150588-97.jpg?t=st=1733903572~exp=1733907172~hmac=4568955ee943dda5e90329788e0a85f775808d2496b52f623d07b9d8b6369eae&w=996",
    "https://img.freepik.com/free-photo/big-head-guy-makes-crazy-face-emotions_144627-3568.jpg?t=st=1733903502~exp=1733907102~hmac=bb8428d7c96a32adf178290ff699d80d62eb960b101dc2e9b6387563a7b1cfc8&w=996",
    "https://img.freepik.com/free-photo/portrait-young-woman-with-natural-make-up_23-2149084942.jpg?t=st=1733585745~exp=1733589345~hmac=d3d00c8a3e774ee78d61dff3735c3aeb423bf2fce43f57126b8126ef82f5a40f&w=826",
    "https://img.freepik.com/free-photo/brunette-young-woman-posing_144627-35773.jpg?t=st=1733586842~exp=1733590442~hmac=44ca65035cfaa31b36c720df62ffb5075826407be34a0028b2315fb990ecb52c&w=900",
    "https://img.freepik.com/premium-photo/beautiful-blonde-caucasian-woman-white-wall_132075-3512.jpg?w=2000",
    "https://img.freepik.com/premium-photo/beautiful-blonde-caucasian-woman-white-wall_132075-3514.jpg?w=2000",
    "https://img.freepik.com/free-photo/worldface-british-guy-white-background_53876-14467.jpg?t=st=1733903291~exp=1733906891~hmac=243898bdf07bf8008a1ab2827ad3106ae7f0bda676e35ddc493c5f625d9e0bc6&w=1380",
    "https://img.freepik.com/free-photo/leisure-activity-people-city-concept-picture-handsome-joyful-young-unshaven-man-black_343059-3676.jpg?ga=GA1.1.63636872.1732969567&semt=ais_hybrid",
    "https://images.pexels.com/photos/1399016/pexels-photo-1399016.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsNWPhXbh68-pBV7iNSR76TAgOVQRSqkuogA&s",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpP8K2NTq306Vg2KKj-fo0BP6n4YoilC6RIw&s",
    "https://www.shutterstock.com/image-photo/vertical-close-portrait-ethnic-young-260nw-2255849067.jpg",
    "https://www.shutterstock.com/image-photo/portrait-real-black-african-woman-260nw-505290286.jpg"
]

process_images(image_urls)

