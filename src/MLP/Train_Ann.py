from Neural_Network import Neural_Network


def train_network():
    ann = Neural_Network()
    datasetTrain, datasetTest = ann.load_data(
        r'C:\Users\Raphael Nascimento\PycharmProjects\skin-melanoma-detector\src\Data\Dataset_ISIC_Normalizado.csv', 165)
    xTrain = datasetTrain[:, 1:10]
    yTrain = datasetTrain[:, 0]

    model = ann.createModel()
    history = ann.trainModel(model, xTrain, yTrain, 200)
    print(ann.evaluateModel(model, xTrain, yTrain))

    xTest = datasetTest[:, 1:10]
    yTest = datasetTest[:, 0]
    print(ann.evaluateModel(model, xTest, yTest))

    predictions = ann.makePredictions_classes(model, xTest)

    correct = 0
    incorrect = 0
    totalSpec = 0
    totalSensi = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in yTest:
        if i == 0:
            totalSpec += 1
        else:
            totalSensi += 1

    for i in range(len(xTest)):
        if predictions[i] == yTest[i]:
            correct += 1
            if predictions[i] == 0:
                TN += 1
            elif predictions[i] == 1:
                TP += 1
        else:
            incorrect += 1
            if predictions[i] == 0:
                FN += 1
            elif predictions[i] == 1:
                FP += 1

    print('''Corretas: %s; 
Incorretas: %s; 
Especificidade: %s; 
Sensibilidade: %s;
    ''' %
          (correct, incorrect, TN, TP))

    print('Total Especificidade: %s; Total Sensibilidade: %s;' % (totalSpec, totalSensi))

    print('''Porcentagem Especificidade: %s; 
Porcentagem Sensibilidade: %s; 
Porcentagem Acurácia: %s;
    ''' %
          (TN / totalSpec * 100, TP / totalSensi * 100, correct / len(xTest) * 100))
    print('Especificidade: %s' % ((TN / (TN + FP)) * 100))
    print('Sensibilidade: %s' % ((TP / (TP + FN)) * 100))
    print('Precisão: %s' % ((TP / (TP + FP)) * 100))
    print('Acurácia: %s' % (((TP + TN) / (TP + TN + FP + FN)) * 100))

    predictions_ravel = ann.make_predictions_ravel(model, xTest)
    ann.roc_curve(predictions_ravel, yTest)

    return ann, model


def save_ann(ann, model):
    save = int(input("Salvar a Rede: "))
    if save == 1:
        print(ann.export_model(model))
    else:
        print('Finish')


if __name__ == '__main__':
    ann, model = train_network()
    save_ann(ann, model)
