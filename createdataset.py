import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Boolean, MetaData, Table
from sqlalchemy.pool import NullPool

import os
import tkinter.ttk as ttk
import tkinter.font as font
import shutil
import tkinter
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from trainer import trainer
from videotester import detector
from facialemotion import emotion


def dataset():  # Function for Creating Dataset..
    window = tkinter.Toplevel()
    window.geometry("1380x860")

    window.title("Face Recognition And Emotion Detection")

    message = Label(window, text="Face Recognition And Emotion Detection", font=(
        "arial black", 25), fg="#1261A0", bg="#F6F6F6", height=2, width=45)
    message.place(x=180, y=70)

    lbl = tk.Label(window, text="MENU", width=12, height=1,
                   fg='#1261A0', font=("arial black", 18))
    lbl.place(x=200, y=220)

    btn2 = Button(window, text="TRAIN FACE DATASET", font=("arial Black", 12),
                  activebackground='#0099cc', fg='#1261A0', command=trainer, width=18, height=1)
    btn2.place(x=200, y=300)

    btn3 = Button(window, text="TRAIN EMOTION DATASET", font=('arial Black', 12),
                  activebackground='#0099cc', fg='#1261A0', command=emotion, width=22, height=1)
    btn3.place(x=200, y=350)

    btn4 = Button(window, text="DETECT EMOTIONS", font=('arial Black', 12),
                  activebackground='#0099cc', fg='#1261A0', command=detector, width=18, height=1)
    btn4.place(x=200, y=400)

    btn7 = Button(window, text="Back", font=('arial Black', 12),
                  activebackground='#0099cc', fg='#1261A0', command=window.destroy, width=18, height=1)
    btn7.place(x=200, y=450)
    my_canvas = Canvas(window, width=3, height=450)
    my_canvas.place(x=500, y=220)
    my_canvas.create_line(4, 0, 4, 450, fill='black')
    lbl = tk.Label(window, text="ID", width=12, height=1,
                   fg='#1261A0', font=("arial black", 12))
    lbl.place(x=600, y=220)
    txt = tk.Entry(window, width=20, bg='white',
                   fg='black', font=("times new roman", 12))
    txt.pack()
    txt.place(x=800, y=220)

    lbl2 = tk.Label(window, text="NAME", width=12, height=1,
                    fg='#1261A0', font=('arial black', 12))
    lbl2.place(x=600, y=270)
    txt2 = tk.Entry(window, width=20, bg='white',
                    fg='black', font=('times new roman', 12))
    txt2.pack()
    txt2.place(x=800, y=270)

    lbl6 = tk.Label(window, text="NOTIFICATION", width=12,
                    height=1, fg='#1261A0', font=('arial black', 12))
    lbl6.place(x=600, y=320)
    message2 = tk.Label(window, text='', bg='white', fg='black',
                        width=60, height=1, font=('times new roman', 12))
    message2.place(x=800, y=320)

    def clear():  # Function for Clear the Fields
        txt.delete(0, 'end')
        txt2.delete(0, 'end')
        res = ""
        message2.configure(res)

    def createdataset():  # Function for Take Images and Store Student Record in Csv File And In Database Only Record
        Id = txt.get()
        Name = txt2.get()

        faceDetect = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')
        cam1 = cv2.VideoCapture(0)
        cam2 = cv2.VideoCapture(2)
        samplenum = 0
        while True:
            ret1, test_img1 = cam1.read()
            ret2, test_img2 = cam2.read()
            gray_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
            faces_1 = faceDetect.detectMultiScale(gray_img1, 1.3, 3)
            gray_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
            faces_2 = faceDetect.detectMultiScale(gray_img2, 1.3, 3)
            for(x, y, w, h) in faces_1:
                samplenum = samplenum+1
                cv2.imwrite('./dataSet/Student.'+str(Id)+'.' +
                            str(samplenum)+'.jpg', gray_img1[y:y+h, x:x+w])
                cv2.putText(test_img1, str(samplenum), (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(test_img1, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for(x, y, w, h) in faces_2:
                samplenum = samplenum+1
                cv2.imwrite('./dataSet/Student.'+str(Id)+'.' +
                            str(samplenum)+'.jpg', gray_img2[y:y+h, x:x+w])
                cv2.putText(test_img2, str(samplenum), (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(test_img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if(ret1):
                cv2.imshow('cam1', test_img1)
            if(ret2):
                cv2.imshow('cam2', test_img2)
            if cv2.waitKey(1) == 13:
                break
            elif samplenum > 200:
                break

        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()
        res = 'Images saved for ID:'+Id + 'Name:'+Name
        row = [Id, Name]
        message2.configure(text=res)

        # Function for Inster Student Record In database
        def insertorupdate(Id, Name):
            engine = create_engine(
                'mssql+pyodbc://LAPTOP-AO2VFMDV\DIPESH/FaceRecognition?driver=SQL+Server+Native+Client+11.0', echo=True, poolclass=NullPool)


                #'mssql+pyodbc://@LAPTOP-AO2VFMDV/DIPESH//FaceRecognition?driver=SQL+Server+Native+Client+11.0', echo=True, poolclass=NullPool)
            meta = MetaData()
            Face_Emotions = Table(
                'Face_Emotions', meta,
                Column('Id', Integer, primary_key=True),
                Column('Name', String),

            )
            meta.create_all(engine)
            ins = Face_Emotions.insert()
            ins = Face_Emotions.insert().values(Id=Id, Name=Name)
            conn = engine.connect()
            result = conn.execute(ins)

        insertorupdate(Id, Name)

    takeimage = tk.Button(window, text='TAKE IMAGES', command=createdataset, fg='#1261A0',
                          width=15, height=1, activebackground='#0099cc', font=('arial black', 12))
    takeimage.pack()
    takeimage.place(x=600, y=450)

    clearbutton = tk.Button(window, text='CLEAR FIELDS', command=clear, fg='#1261A0',
                            width=15, height=1, activebackground='#0099cc', font=('arial black', 12))
    clearbutton.place(x=800, y=450)

    window.mainloop()



