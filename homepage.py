import tkinter
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import *
import os
from createdataset import dataset
from trainer import trainer
from videotester import detector
from facialemotion import emotion


def homepage():

    window = tkinter.Tk()

    window.title("Face Recognition And Emotion Detection")
    window.geometry("700x860")

    message = Label(window, text="Face Recognition And Emotion Detection", font=(
        "arial black", 20), fg="#1261A0", bg="#F6F6F6", height=1, width=33)
    message.place(x=40, y=70)

    lbl = Label(window, text="MENU", width=12, height=1,
                fg='#1261A0', bg="#F6F6F6", font=("arial black", 18))
    lbl.place(x=200, y=230)
    btn1 = Button(window, text="CREATE DATASET", font=("arial Black", 12),
                  activebackground='#0099cc', fg='#1261A0', command=dataset, width=18, height=1)
    btn1.place(x=200, y=300)

    btn2 = Button(window, text="TRAIN FACE DATASET", font=("arial Black", 12),
                  activebackground='#0099cc', fg='#1261A0', command=trainer, width=18, height=1)
    btn2.place(x=200, y=350)

    btn3 = Button(window, text="TRAIN EMOTION ", font=("arial Black", 12),
                  activebackground='#0099cc', fg='#1261A0', command=emotion, width=18, height=1)
    btn3.place(x=200, y=400)

    btn4 = Button(window, text="DETECT EMOTIONS", font=('arial Black', 12),
                  activebackground='#0099cc', fg='#1261A0', command=detector, width=18, height=1)
    btn4.place(x=200, y=450)

    btn7 = Button(window, text="EXIT", font=('arial Black', 12), activebackground='#0099cc',
                  fg='#1261A0', command=window.destroy, width=18, height=1)
    btn7.place(x=200, y=500)
    window.mainloop()
homepage()
