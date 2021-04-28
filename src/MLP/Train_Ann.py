from Neural_Network import Neural_Network
import matplotlib.pyplot as plt


def train_network():
    ann = Neural_Network()
    datasetTrain, datasetTest = ann.load_data(
        r'Path Dataset')
    xTrain = datasetTrain[:, 1:10]
    yTrain = datasetTrain[:, 0]

    model = ann.createModel()
    history = ann.trainModel(model, xTrain, yTrain, 160)
    print(ann.evaluateModel(model, xTrain, yTrain))

    xTest = datasetTest[:, 1:10]
    yTest = datasetTest[:, 0]
    print(ann.evaluateModel(model, xTest, yTest))

    predictions = ann.makePredictions(model, xTest)
    correct = 0
    incorrect = 0
    specificidade = 0
    sensibilidade = 0
    totalSpecificidade = 0
    totalSensibilidade = 0

    for i in yTest:
        if i == 0:
            totalSpecificidade += 1
        else:
            totalSensibilidade += 1

    for i in range(len(xTest)):
        if predictions[i] == yTest[i]:
            correct += 1
            if predictions[i] == yTest[i] and predictions[i] == 0:
                specificidade += 1
            elif predictions[i] == yTest[i] and predictions[i] == 1:
                sensibilidade += 1
        else:
            incorrect += 1

    # print('%s => %d (expected %d)' % (i, predictions[i], yTrain[i]))
    print('Corretas: %.2f \nIncorretas: %.2f \nEspecificidade: %.2f \nSensibilidade: %.2f \n' %
          ((correct), (incorrect), (specificidade), (sensibilidade)))
    print('Total Especificidade: %s \nTotal Sensibilidade: %s \n' %
          (totalSpecificidade, totalSensibilidade))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acuracia do Modelo')
    plt.ylabel('Acuracia')
    plt.xlabel('Epoca')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perca do Modelo')
    plt.ylabel('Perca')
    plt.xlabel('Epoca')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    plt.show()

    return (ann, model)


def save_ann(ann, model):
    salvar = int(input("Salvar a Rede: "))
    if salvar == 1:
        print(ann.export_model(model))
    else:
        print('Finish')


if __name__ == '__main__':
    ann, model = train_network()
    save_ann(ann, model)
