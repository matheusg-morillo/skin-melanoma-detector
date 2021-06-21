import numpy
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

numpy.random.seed(2324)

dataset = numpy.loadtxt(
  r"C:\Users\Raphael Nascimento\PycharmProjects\skin-melanoma-detector\src\Data\Dataset_ISIC_Normalizado.csv", delimiter=",")

# split into input (X) and output (Y) variables
indices = numpy.random.choice(dataset.shape[0], 60, replace=False)
dataset2 = dataset[indices]
dataset = numpy.delete(dataset, indices, axis=0)

x = dataset[:,1:10]
y = dataset[:,0]

Xtest = dataset2[:,1:10]
Ytest = dataset2[:,0]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)

predictions = knn.predict(Xtest)
predictions2 = knn.predict_proba(Xtest)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Ytest, predictions2[:, 1])
auc_KNN = auc(false_positive_rate1, true_positive_rate1)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(false_positive_rate1, true_positive_rate1, label='KNN - PH2 (area = {:.3f})'.format(auc_KNN))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

correct = 0
incorrect = 0
totalSpec = 0
totalSensi = 0
TP = 0
FP = 0
TN = 0
FN = 0

for i in Ytest:
  if i == 0:
    totalSpec += 1
  else:
    totalSensi += 1

for i in range(len(Xtest)):
  if predictions[i] == Ytest[i]:
    correct += 1
    if predictions[i] == 0:
      TN += 1
    elif predictions[i] == 1:
      TP += 1
  else:
    incorrect +=1
    if predictions[i] == 0:
      FN += 1
    elif predictions[i] == 1:
      FP += 1

print('''Corrects: %s; 
Incorrects: %s; 
Specificity: %s; 
Sensitivity: %s;
''' %
(correct , incorrect, TN, TP))

print('\nTotal TN: %s; Total TP: %s;' % (totalSpec, totalSensi))

print('Specificity: %s' % ((TN / (TN + FP)) * 100))
print('Sensitivity: %s' % ((TP / (TP + FN)) * 100))
print('Precision: %s' % ((TP / (TP + FP)) * 100))
print('Accuracy: %s' % (((TP + TN) / (TP + TN + FP + FN)) * 100))

knnPickle = open('Modelo_ISIC_KNN', 'wb')
pickle.dump(knn, knnPickle)