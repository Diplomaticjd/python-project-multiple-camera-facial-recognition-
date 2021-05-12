import os
import cv2
import numpy as np
from PIL import Image


def trainer():  # Function for Train Dataset
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    path = './dataSet'

    def getImagewithID(path):  # Function for Assign Labels to Dataset
        try:
            imagepath = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            id = []
            for imagepath in imagepath:
                faceImg = Image.open(imagepath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagepath)[-1].split('.')[1])
                faces.append(faceNp)
                id.append(ID)
                cv2.imshow('training', faceNp)
                cv2.waitKey(10)
            return np.array(id), faces
        except:
            print("Images must be contain Id")
    id, faces = getImagewithID(path)

    recognizer.train(faces, id)
    recognizer.save('./recognizer/trainningdata.yml')
    cv2.destroyAllWindows()



