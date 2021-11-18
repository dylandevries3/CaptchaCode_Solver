import cv2
import pickle
import os.path
import numpy
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

#images and answers array
images = []
answers = []

for base_image in paths.list_images("letters"):
    #grayscale and resize
    image = cv2.imread(base_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    image = numpy.expand_dims(image, axis=2)

    #get answer from file
    answer = base_image.split(os.path.sep)[-2]

    images.append(image)
    answers.append(answer)


images = numpy.array(images, dtype="float") / 255.0
answers = numpy.array(answers)

#train data and validation data
(X_train, X_test, Y_train, Y_test) = train_test_split(images, answers, test_size=0.25, random_state=0)

map = LabelBinarizer().fit(Y_train)
Y_train = map.transform(Y_train)
Y_test = map.transform(Y_test)

#output map
with open("model_labels.dat", "wb") as mapped:
    pickle.dump(map, mapped)


model = Sequential()
#conv1
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#conv2
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#hidden layer
model.add(Flatten())
model.add(Dense(100, activation="relu"))

#dense layer
model.add(Dense(32, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

model.save("model_final.hdf5")
