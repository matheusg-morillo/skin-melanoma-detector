from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np


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
        return cv2.threshold(imgBlur, initialValue, limitValue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def binaryImg2(self, imgBlur, initialValue, limitValue):
        return cv2.threshold(imgBlur, initialValue, limitValue, cv2.THRESH_BINARY_INV)

    def removeSmallAreas(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.maxArea(contours):
                cv2.drawContours(img, [c], -1, (0, 0, 0), -1)

    def maxArea(self, contours):
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
        return max_area


    def morphologyClose(self, binaryImg, structureElement, iterations):
        return cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT,
                                                          structureElement), iterations=iterations)

    def morphologyOpen(self, binaryImg, structureElement):
        return cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT,
                                                          structureElement))

    def canny(self, bynaryImg):
        return cv2.Canny(bynaryImg, 0, 255)

    def resizeImg(self, img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def convertToBMP(self, pathImg, pathImgSave):
        Image.open(pathImg).save(pathImgSave)

    def histogram(self, img):
        return cv2.calcHist([img], [0], None, [256], [0, 256])

    def plotHist(self, histogram):
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Intensidade")
        plt.ylabel("Quantidade de Pixels")
        plt.plot(histogram)
        plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(0)

    def erosion(self, binaryIMG, iterations):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.erode(binaryIMG, kernel, iterations=iterations)

    def dilation(self, binaryIMG, iterations):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(binaryIMG, kernel, iterations=iterations)

    def removeHair(self, img, structureElement):
        (b, g, r) = cv2.split(img)
        b2 = self.morphologyClose(b, structureElement, 1)
        g2 = self.morphologyClose(g, structureElement, 1)
        r2 = self.morphologyClose(r, structureElement, 1)
        return cv2.merge((b2, g2, r2))
