import csv


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
                    return str(row["benign_malignant"])

    def writeCSV(self, imageName, benign_malignant, descriptors):
        with open(self.path + '\\' + 'Descritores.csv',
                  'a', newline='') as csv_file:

            fieldnames = ["benign_malignant", "image", "simmetryX",
                          "simmetryY", "radius", "meanR", "varianceR", "meanG",
                          "varianceG", "meanB", "varianceB"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({"benign_malignant": benign_malignant,
                             "image": imageName,
                             "simmetryX": descriptors[0],
                             "simmetryY": descriptors[1],
                             "radius": descriptors[2],
                             "meanR": descriptors[3],
                             "varianceR": descriptors[4],
                             "meanG": descriptors[5],
                             "varianceG": descriptors[6],
                             "meanB": descriptors[7],
                             "varianceB": descriptors[8]})
