from PIL import Image
from matplotlib import pyplot as plt
import cv2


class PreProcessing_Module(object):

    def openImg(self, pathImg):
        return cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)

    def bgr2Gray(self, imageBgr):
        return cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)

    def equalizeImg(self, imgGray):
        return cv2.equalizeHist(imgGray)

    def gaussianBlur(self, imgGray, gaussianValue):
        return cv2.GaussianBlur(imgGray, gaussianValue, 0)

    def binaryImg(self, imgBlur, initialValue, limitValue):
        return cv2.threshold(imgBlur, initialValue, limitValue,
                             cv2.THRESH_BINARY_INV)

    def morphologyClose(self, binaryImg):
        return cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT,
                                                          (5, 5)))

    def canny(self, bynaryImg):
        return cv2.Canny(bynaryImg, 0, 255)

    def resizeImg(self, img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def convertToBMP(self, pathImg, pathImgSave):
        Image.open(pathImg).save(pathImgSave)

    def histograma(self, img):
        return cv2.calcHist([img], [0], None, [256], [0, 256])

    def plotHist(self, histograma):
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Intensidade")
        plt.ylabel("Quantidade de Pixels")
        plt.plot(histograma)
        plt.xlim([0, 256])
        plt.show()
        cv2.waitkey(0)
