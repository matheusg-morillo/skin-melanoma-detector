import csv
import pandas as pd


class Data_Extrator(object):
    def __init__(self, path):
        self.path = path

    def readCSV(self, pathCSV, imageName):
        with open(pathCSV + '\\' + 'melanoma_dados.csv') as csv_file:
            csv_reader = csv.DictReader(
                csv_file, fieldnames=["id", "name", "benign_malignant",
                                      "diagnosis"])

            csv_reader.__next__()

            for row in csv_reader:
                if row["name"] == imageName:
                    if row["benign_malignant"] == 'benign':
                        return 0
                    else:
                        return 1

    def readTXT(self, imageName):
        df = pd.read_csv(
            r'C:\Users\rapha\Documents\PH_Database\PH2_dataset.txt', delimiter='|')
        for i in range(len(df)):
            if df.loc[i, "   Name "] == imageName:
               if df.loc[i, " Clinical Diagnosis "] == 0 or df.loc[i, " Clinical Diagnosis "] == 1:
                   return 0.0
               else:
                   return 1.0

    def writeCSV(self, benign_malignant, descriptors):
        with open(self.path + 'Dataset_ImagensASIC.csv',
                  'a', newline='') as csv_file:

            fieldnames = ["benign_malignant", "simmetryX",
                          "simmetryY", "radius", "meanR", "varianceR", "meanG",
                          "varianceG", "meanB", "varianceB"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({"benign_malignant": benign_malignant,
                             "simmetryX": descriptors[0],
                             "simmetryY": descriptors[1],
                             "radius": descriptors[2],
                             "meanR": descriptors[3],
                             "varianceR": descriptors[4],
                             "meanG": descriptors[5],
                             "varianceG": descriptors[6],
                             "meanB": descriptors[7],
                             "varianceB": descriptors[8]})

    def normalize(self):
        df = pd.read_csv(self.path + '\\Dataset_ImagensASIC.csv')
        df2 = (df - df.min()) / (df.max() - df.min())
        df2.to_csv(self.path + '\\Dataset_ImagensASIC_Normalizado2.csv', index=False)

    def imageName(self, imgProcessed, imgNotProcessed):
        with open(self.path + 'imagens.csv',
                  'w', newline='') as csv_file:
            fieldnames = ["imageProcessed", "imageNotProcessed"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for nameImgProcessed in imgProcessed:
                writer.writerow({"imageProcessed": nameImgProcessed,
                                 "imageNotProcessed": 0})
            for nameImgNotProcessed in imgNotProcessed:
                writer.writerow({"imageProcessed": 0,
                                 "imageNotProcessed": nameImgNotProcessed})
