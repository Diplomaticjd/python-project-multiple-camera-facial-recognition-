import sys
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

def emotion():
    # read dataset and store in dataframe
    df = pd.read_csv("./fer2011.csv")
    # intialize 4 lists
    X_train, train_Y, X_test, test_Y = [], [], [], []
    # iterate our dataset
    for index, row in df.iterrows():
        # split dataset on pixels
        val = row["pixels"].split(" ")
        try:
            if "Training" in row["Usage"]:
                # append pixels in X_train list and empotion in train_Y
                X_train.append(np.array(val, "float32"))
                train_Y.append(row["emotion"])
            elif "PublicTest" in row["Usage"]:
                # append pixels in X_test list and empotion in test_Y
                X_test.append(np.array(val, "float32"))
                test_Y.append(row["emotion"])
        except:
            print(f"error occurd at index:{index} and row:{row}")

    # covert lists in numpy array
    X_train = np.array(X_train, "float32")
    train_Y = np.array(train_Y, "float32")
    X_test = np.array(X_test, "float32")
    test_Y = np.array(test_Y, "float32")

    # Normalizing data between 0 and 1

    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)

    num_features = 64  # number of features from each data
    num_labels = 7     # number of labels
    batch_size = 64    # batch size
    epochs = 30        # Epochs
    width, height = 48, 48  # width , height

    X_train = X_train.reshape(X_train.shape[0], width, height, 1)  # reshape
    X_test = X_test.reshape(X_test.shape[0], width, height, 1)  # reshape
    train_Y = np_utils.to_categorical(train_Y, num_classes=num_labels)
    test_Y = np_utils.to_categorical(test_Y, num_classes=num_labels)


    #designing in cnn

    model = Sequential()
    # 1st layer
    model.add(Conv2D(num_features, kernel_size=(3, 3),
                    activation="relu", input_shape=(X_train.shape[1:])))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd layer
    model.add(Conv2D(num_features, (3, 3), activation="relu"))
    model.add(Conv2D(num_features, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd layer
    model.add(Conv2D(2*num_features, (3, 3), activation="relu"))
    model.add(Conv2D(2*num_features, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2*2*2*2*num_features, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2*2*2*2*num_features, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                optimizer=adam(), metrics=["accuracy"])
    model.fit(X_train, train_Y, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(X_test, test_Y), shuffle=True)

    # saving model
    fer_json = model.to_json()
    with open("./fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("./fer.h5")
    
