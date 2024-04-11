import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

class DenseNetClassifier:

    def __init__(self, labels):
        self.labels = LabelEncoder()
        self.labels.fit(labels)

        model = model = Sequential()
        model.add(Dense(512, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(len(self.labels.classes_), activation="softmax"))

        opt = Adam(lr=0.000001)
        model.compile(
            optimizer = opt,
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'])
        
        self.model = model

    def fit(self, visages, noms):
        labels = self.labels.transform(noms)
        print('Visage', visages.shape)
        print('Labels', labels.shape)
        history = self.model.fit(
            visages, labels,
            epochs = 500)

    def predict(self, visage):
        pred = self.model.predict(visage)
        return self.labels.inverse_transform(np.argmax(pred, axis=-1))


class ConvNetClassifier:

    def __init__(self, labels):
        self.labels = LabelEncoder()
        self.labels.fit(labels)

        model = model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(50, 50, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(16, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        # model.add(Conv2D(64, 3, padding="same", activation="relu"))
        # model.add(MaxPool2D())
        # model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(len(self.labels.classes_), activation="softmax"))

        opt = Adam(lr=0.000001)
        model.compile(
            optimizer = opt,
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'])
        
        self.model = model

    def fit(self, visages, noms):
        labels = self.labels.transform(noms)
        print('Visage', visages.shape)
        print('Labels', labels.shape)
        history = self.model.fit(
            visages, labels,
            epochs = 500)

    def predict(self, visage):
        pred = self.model.predict(visage)
        return self.labels.inverse_transform(np.argmax(pred, axis=-1))
