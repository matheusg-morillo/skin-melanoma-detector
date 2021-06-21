from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


class Neural_Network(object):
    numpy.random.seed(2324)

    def load_data(self, pathDataset, samples):
        datasetTrain = numpy.loadtxt(pathDataset, delimiter=",")
        indices = numpy.random.choice(datasetTrain.shape[0], samples, replace=False)
        datasetTest = datasetTrain[indices]
        datasetTrain = numpy.delete(datasetTrain, indices, axis=0)
        return (datasetTrain, datasetTest)

    def createModel(self):
        model = Sequential()
        model.add(Dense(12, input_dim=9, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def trainModel(self, model, X, Y, epochs):
        history = model.fit(X, Y, epochs=epochs, batch_size=10, verbose=0)
        return history

    def evaluateModel(self, model, X, Y):
        _, accuracy = model.evaluate(X, Y)
        return 'Accuracy: %.2f' % (accuracy * 100)

    def makePredictions_classes(self, model, X):
        return model.predict_classes(X)

    def make_predictions_ravel(self, model, X):
        return model.predict(X).ravel()

    def roc_curve(self, predictions, Y):
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y, predictions)
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='RMSprop - PH2 (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    def export_model(self, model):
        model_to_json = model.to_json()
        with open("../../Modelo_PH2/model_ISIC.json", "w") as json_file:
            json_file.write(model_to_json)
        model.save_weights("weights_ISIC.h5")
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
