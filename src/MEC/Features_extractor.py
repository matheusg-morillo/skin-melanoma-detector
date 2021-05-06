import numpy as np
import cv2


class Features_Extractor(object):

    def findContours(self, cannyImg):
        return cv2.findContours(cannyImg, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

    def findMoments(self, contours):
        return cv2.moments(contours)

    def findCenterOfMass(self, moments):
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)

    def findBinaryPixels(self, binaryImg, rows, cols):
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
        return abs(x1 - x2)

    def findSymmetryY(self, binaryImg, rows, cols, cY):
        y1 = y2 = 0
        for i in range(rows):
            for j in range(cols):
                if j < cY:
                    y1 += binaryImg[i, j]
                else:
                    y2 += binaryImg[i, j]
        return abs(y1 - y2)

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
                blueVariance += (int(b) - blueMean)**2 * int(binaryImg[i, j])
                greenVariance += (int(g) - greenMean)**2 * int(binaryImg[i, j])
                redVariance += (int(r) - redMean)**2 * int(binaryImg[i, j])
        blueVariance = (1/binaryPixels) * blueVariance
        greenVariance = (1/binaryPixels) * greenVariance
        redVariance = (1/binaryPixels) * redVariance
        return (blueVariance, greenVariance, redVariance)

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
        return cv2.circle(img, coordinates, int(radius), (0, 255, 0), 3)

    def drawContours(self, img, coordinates, index):
        return cv2.drawContours(img, coordinates, index, (0, 255, 0), 3)

    def save(self, nameIMG, img):
        cv2.imwrite(nameIMG, img)
