import cv2
import os
import numpy as np
import streamlit as st
import winsound

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

st.title("Facial Recognition using ANN")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Capture Dataset
# -------------------------------

name = st.text_input("Enter Person Name")

if st.button("Capture Faces"):

    person_path = os.path.join(dataset_path, name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0
    max_images = 30

    while True:

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face,(64,64))

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,
                        f"Press S to capture | Captured: {count}",
                        (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

            key = cv2.waitKey(1)

            if key == ord('s'):

                img_path = f"{person_path}/{count}.jpg"

                cv2.imwrite(img_path, face)

                winsound.Beep(1000,200)

                count += 1

        cv2.imshow("Face Capture", frame)

        if count >= max_images:

            print("Dataset collection completed")

            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    st.success(f"{count} images captured successfully")

# -------------------------------
# Load Dataset
# -------------------------------

def load_dataset():

    images = []
    labels = []

    for person in os.listdir(dataset_path):

        person_folder = os.path.join(dataset_path, person)

        for img_name in os.listdir(person_folder):

            img_path = os.path.join(person_folder, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img,(64,64))

            images.append(img)

            labels.append(person)

    images = np.array(images)/255.0
    images = images.reshape(len(images),64,64,1)

    labels = np.array(labels)

    return images, labels

# -------------------------------
# Train Model
# -------------------------------

if st.button("Train Model"):

    images, labels = load_dataset()

    encoder = LabelEncoder()

    labels = encoder.fit_transform(labels)

    labels = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2
    )

    model = Sequential()

    model.add(Flatten(input_shape=(64,64,1)))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(len(labels[0]), activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=15)

    loss, accuracy = model.evaluate(X_test, y_test)

    st.success(f"Model Accuracy: {accuracy*100:.2f}%")

    model.save("face_model.h5")

# -------------------------------
# Real-time Recognition
# -------------------------------

if st.button("Start Face Recognition"):

    model = load_model("face_model.h5")

    images, labels = load_dataset()

    encoder = LabelEncoder()
    encoder.fit(labels)

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face,(64,64))

            face = face/255.0

            face = face.reshape(1,64,64,1)

            prediction = model.predict(face)

            index = np.argmax(prediction)

            name = encoder.inverse_transform([index])[0]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

        cv2.imshow("Face Recognition",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()