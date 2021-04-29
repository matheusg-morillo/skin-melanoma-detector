# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy


class Neural_Network(object):

    def load_data(self, pathDataset):
        datasetTrain = numpy.loadtxt(pathDataset, delimiter=",")
        indices = numpy.random.choice(datasetTrain.shape[0], 30, replace=False)
        datasetTest = datasetTrain[indices]
        datasetTrain = numpy.delete(datasetTrain, indices, axis=0)
        return (datasetTrain, datasetTest)

    def createModel(self):
        model = Sequential()
        model.add(Dense(10, input_dim=9, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        return model

    def trainModel(self, model, X, Y, epochs):
        history = model.fit(X, Y, epochs=epochs, batch_size=10, validation_split=0.33, verbose=0)
        return history

    def evaluateModel(self, model, X, Y):
        _, accuracy = model.evaluate(X, Y)
        return 'Accuracy: %.2f' % (accuracy * 100)

    def makePredictions(self, model, X):
        predictions = model.predict_classes(X)
        return predictions

    def export_model(self, model):
        model_to_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_to_json)
        model.save_weights("weights.h5")
        return 'Model Exported'

    def load_model(self, pathModel, pathWeights):
        json_model = open(pathModel, 'r')
        model = json_model.read()
        json_model.close()
        model = model_from_json(model)
        model.load_weights(pathWeights)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])
        return model
