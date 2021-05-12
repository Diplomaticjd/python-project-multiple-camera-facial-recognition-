import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Boolean, MetaData, Table
from sqlalchemy.pool import NullPool
from datetime import datetime
def detector():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(
        './recognizer/trainningdata.yml')
    path = 'D:/project/dataSet'
    # load model
    model = model_from_json(open("./fer.json", "r").read())
    # load weights
    model.load_weights('./fer.h5')

    # haar_cascade to detect faces
    face_haar_cascade = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')

    # haarcascade_eye detect Eyes
    eye_haar_cascade = cv2.CascadeClassifier(
        './haarcascade_eye.xml')


    def getprofile(Id):  # Function for Get Student Name and RollNo from Database for Matching at time of Recognition
        engine = create_engine(
                'mssql+pyodbc://LAPTOP-AO2VFMDV\DIPESH/FaceRecognition?driver=SQL+Server+Native+Client+11.0', echo=True, poolclass=NullPool)
        meta = MetaData()
        Face_Emotions = Table(
            'Face_Emotions', meta,
            Column('Id', Integer, primary_key=True),
            Column('Name', String),
        )
        s = Face_Emotions.select().where(Face_Emotions.c.Id == id)
        conn = engine.connect()
        cursor = conn.execute(s)
        profile = None
        for row in cursor:
            profile = row
        return profile


    cam1 = cv2.VideoCapture(0)  # for laptop cam
    cam2 = cv2.VideoCapture(2)  # for external web cam

    while True:
        # captures frame and returns boolean value and captured image
        ret1, test_img1 = cam1.read()
        ret2, test_img2 = cam2.read()

        gray_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
        # detect multi scale takes image,scale factor,neighbours
        faces_detected = face_haar_cascade.detectMultiScale(gray_img1, 1.3, 3)
        facescam2_detected = face_haar_cascade.detectMultiScale(
            gray_img2, 1.3, 3)
        for (x, y, w, h) in faces_detected:
            id, conf = recognizer.predict(gray_img1[y:y+h, x:x+w])
            if conf < 55:
                print('Confidence: ', conf)
                print('ID: ', id)
                cv2.rectangle(test_img1, (x, y), (x+w, y+h),
                            (255, 0, 0), thickness=1)
                profile = getprofile(id)
                if profile is not None:

                    # cropping region of interest i.e. face area from  image
                    # preprossing image for model prediction
                    roi_gray = gray_img1[y:y+w, x:x+h]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    predictions = model.predict(img_pixels)

                    # find max indexed array
                    max_index = np.argmax(predictions[0])
                    # number of emotions
                    emotions = ('', '', '', 'happy',
                                'sad', '', 'neutral')
                    # pass value of max_index to emotions
                    predicted_emotion = emotions[max_index]
                    cv2.putText(
                        test_img1, str(profile[0]), (x, y+h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(
                        test_img1, str(profile[1]), (x, y+h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    # puttext is used to display text like happy,sad
                    cv2.putText(test_img1, predicted_emotion, (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for (x, y, w, h) in facescam2_detected:
            id, conf = recognizer.predict(gray_img2[y:y+h, x:x+w])
            if conf < 55:
                print('Confidence: ', conf)
                print('ID: ', id)
                cv2.rectangle(test_img2, (x, y), (x+w, y+h),
                            (255, 0, 0), thickness=1)
                profile = getprofile(id)
                if profile is not None:

                    # cropping region of interest i.e. face area from  image
                    # preprossing image for model prediction
                    roi_gray = gray_img2[y:y+w, x:x+h]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    predictions = model.predict(img_pixels)

                    # find max indexed array
                    max_index = np.argmax(predictions[0])
                    # number of emotions
                    emotions = ('', '', '', 'happy',
                                'sad', '', 'neutral')
                    # pass value of max_index to emotions
                    predicted_emotion = emotions[max_index]
                    cv2.putText(
                        test_img2, str(profile[0]), (x, y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(
                        test_img2, str(profile[1]), (x, y+h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    # puttext is used to display text like happy,sad
                    cv2.putText(test_img2, predicted_emotion, (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # pass image to eye haarcascade
        eyescam1 = eye_haar_cascade.detectMultiScale(gray_img1, 2, 4)
        eyescam2 = eye_haar_cascade.detectMultiScale(gray_img2, 2, 4)
        # rectangle against eyes
        for(x1, y1, w1, h1) in eyescam1:
            cv2.rectangle(test_img1, (x1, y1), (x1+w1, y1+h1),
                        (0, 255, 0), thickness=1)

        for(x1, y1, w1, h1) in eyescam2:
            cv2.rectangle(test_img2, (x1, y1), (x1+w1, y1+h1),
                        (0, 255, 0), thickness=1)

        

        if (ret1):
            cv2.imshow('cam1', test_img1)
        if(ret2):
            cv2.imshow('cam2', test_img2)

        if cv2.waitKey(1) == 13:  # wait until 'q' key is pressed
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows



