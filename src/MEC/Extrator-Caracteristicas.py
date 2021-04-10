import numpy as np
import cv2


class Extrator_Caracteristicas(object):
    def __init__(self, pathImg):
        self.pathImg = pathImg

    def openImg(self):
        return cv2.imread(self.pathImg)

    def bgr2Gray(self, imageBgr):
        return cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)

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

    def findContours(self, cannyImg):
        return cv2.findContours(cannyImg, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

    def findMoments(self, contours):
        return cv2.moments(contours)

    def findCenterOfMass(self, moments):
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)

    def findDescriptors(self, binaryImg, rows, cols):
        binaryPixels = 0
        for i in range(rows):
            for j in range(cols):
                if binaryImg[i, j] == 255:
                    binaryPixels += 1
        return binaryPixels

    def findSymmetryX(self, binaryImg, rows, cols, cX):
        x1 = x2 = 0
        for i in range(rows):
            if i < cX:
                for j in range(cols):
                    x1 += binaryImg[i, j]
            else:
                for j in range(cols):
                    x2 += binaryImg[i, j]
        return x1 - x2

    def findSymmetryY(self, binaryImg, rows, cols, cY):
        y1 = y2 = 0
        for i in range(rows):
            for j in range(cols):
                if j < cY:
                    y1 += binaryImg[i, j]
                else:
                    y2 += binaryImg[i, j]
        return y1 - y2

    def findMinEnclosingCircle(self, coordinates):
        return cv2.minEnclosingCircle(coordinates)

    def findMeanBGR(self, imageBGR, binaryImg, rows, cols, binaryPixels):
        blueMean = greenMean = redMean = 0
        for i in range(rows):
            for j in range(cols):
                (b, g, r) = imageBGR[i, j]
                blueMean += int(b) * int(binaryImg[i, j])
                greenMean += int(g) * int(binaryImg[i, j])
                redMean += int(r) * int(binaryImg[i, j])
        blueMean = (1/binaryPixels) * blueMean
        greenMean = (1/binaryPixels) * greenMean
        redMean = (1/binaryPixels) * redMean
        return (blueMean, greenMean, redMean)

    def findVarianceBGR(self, imageBGR, binaryImg, rows, cols, binaryPixels,
                        blueMean, greenMean, redMean):
        blueVariance = greenVariance = redVariance = 0
        for i in range(rows):
            for j in range(cols):
                (b, g, r) = imageBGR[i, j]
                blueVariance += (b - blueMean)**2 * binaryImg[i, j]
                greenVariance += (g - greenMean)**2 * binaryImg[i, j]
                redMean += (r - redMean)**2 * binaryImg[i, j]
        blueVariance = (1/binaryPixels) * blueVariance
        greenVariance = (1/binaryPixels) * greenVariance
        redVariance = (1/binaryPixels) * redVariance
        return (blueVariance, greenVariance, redMean)

    def showSingleImg(self, img, barText):
        cv2.imshow(barText, img)
        cv2.waitKey(0)

    def show2Img(self, img1, img2, barText):
        imgScreen = np.vstack([np.hstack([img1, img2])])
        cv2.imshow(barText, imgScreen)
        cv2.waitKey(0)

    def show4Img(self, img1, img2, img3, img4, barText):
        imgScreen = np.vstack([np.hstack([img1, img2]),
                               np.hstack([img3, img4])])
        cv2.imshow(barText, imgScreen)
        cv2.waitKey(0)

    def drawCircle(self, img, coordinates, radius):
        return cv2.circle(img, int(coordinates), int(radius), (0, 255, 0), 3)

    def drawContours(self, img, coordinates, index):
        return cv2.drawContours(img, coordinates, index, (0, 255, 0), 3)
