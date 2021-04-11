from MEC import Features_extractor
from MPP import Pre_Processing_Module
from Data import Data_Extrator
import os
import sys


def main():
    mpp = Pre_Processing_Module.PreProcessing_Module()
    mec = Features_extractor.Features_Extractor()
    data = Data_Extrator.Data_Extrator(
        r'C:\Users\rapha\Documents\skin-melanoma-detector\src\Data\\')

    path = r'C:\Users\rapha\Documents\ISIC-download'
    descriptors = []
    index = 0
    initialValue = 125
    blur = (9, 9)

    for arquivo in os.listdir(path):
        imgMask = imgBGR = mpp.openImg(path + '\\' + str(arquivo))
        grayImg = mpp.bgr2Gray(imgMask)
        # grayImg = mpp.equalizeImg(grayImg)
        gaussianBlurImg = mpp.gaussianBlur(grayImg, blur)
        (T, binaryImg) = mpp.binaryImg(gaussianBlurImg, initialValue, 255)
        binaryImg = mpp.morphologyClose(binaryImg)
        borderedImg = mpp.canny(binaryImg)
        contours, hierarchy = mec.findContours(borderedImg)
        moments = mec.findMoments(binaryImg)
        (cX, cY) = mec.findCenterOfMass(moments)
        (rows, cols) = grayImg.shape
        binaryPixels = mec.findBinaryPixels(binaryImg, rows, cols)
        symmetryX = mec.findSymmetryX(binaryImg, rows, cols, cX)
        symmetryY = mec.findSymmetryY(binaryImg, rows, cols, cY)
        (x, y), radius = mec.findMinEnclosingCircle(contours[index])
        (blueMean, greenMean, redMean) = mec.findMeanBGR(imgBGR, binaryImg,
                                                         rows,
                                                         cols, binaryPixels)
        (blueVariance, greenVariance, redVariance) = mec.findVarianceBGR(
            imgBGR, binaryImg, rows, cols, binaryPixels, blueMean, greenMean,
            redMean)

        descriptors.append(symmetryX)
        descriptors.append(symmetryY)
        descriptors.append(radius)
        descriptors.append(redMean)
        descriptors.append(redVariance)
        descriptors.append(greenMean)
        descriptors.append(greenVariance)
        descriptors.append(blueMean)
        descriptors.append(blueVariance)

        print(descriptors)

        grayImg = mec.drawContours(grayImg, contours, -1)
        grayImg = mec.drawCircle(grayImg, (int(cX), int(cY)), 3)
        grayImg = mec.drawCircle(grayImg, (int(x), int(y)), int(radius))
        mec.show2Img(binaryImg, grayImg, 'Modulo Extrator de Caracteristicas')

        print('Size: %s' % (len(contours)))
        index = int(input("Contorno: "))
        if index != -1:
            benign_malignant = data.readCSV(path, arquivo)
            data.writeCSV(str(arquivo), benign_malignant, descriptors)
        else:
            sys.exit()

        descriptors.clear()


if __name__ == "__main__":
    main()
