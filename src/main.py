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

    pathSegmentation = r'C:\Users\rapha\Documents\ISIIC_Download_Segmentation'
    pathImgBGR = r'C:\Users\rapha\Documents\ISIC-download'
    descriptors = []
    index = 0
    imgNotProcessed = []
    imgProcessed = []
    count = 0

    for arquivo in os.listdir(
            r'C:\Users\rapha\Documents\ISIIC_Download_Segmentation'):
        count += 1
        nameImg = arquivo.split('.')
        binaryImg = mpp.openImg(
            pathSegmentation + '\\' + nameImg[0] + '.png')
        imgBGR = mpp.openImg(pathImgBGR + '\\' + nameImg[0] + '.jpg')
        # grayImg = mpp.bgr2Gray(binaryImg)
        # grayImg = mpp.equalizeImg(grayImg)
        # gaussianBlurImg = mpp.gaussianBlur(grayImg, blur)
        # (T, binaryImg) = mpp.binaryImg(grayImg, 120, 255)
        # binaryImg = mpp.morphologyClose(binaryImg)
        # borderedImg = mpp.canny(binaryImg)
        contours, hierarchy = mec.findContours(binaryImg)
        if len(contours) == 1:
            moments = mec.findMoments(binaryImg)
            (cX, cY) = mec.findCenterOfMass(moments)
            (rows, cols) = binaryImg.shape
            binaryPixels = mec.findBinaryPixels(binaryImg, rows, cols)
            symmetryX = mec.findSymmetryX(binaryImg, rows, cols, cX)
            symmetryY = mec.findSymmetryY(binaryImg, rows, cols, cY)
            (x, y), radius = mec.findMinEnclosingCircle(contours[index])
            (blueMean, greenMean, redMean) = mec.findMeanBGR(imgBGR, binaryImg,
                                                             rows,
                                                             cols,
                                                             binaryPixels)
            (blueVariance, greenVariance, redVariance) = mec.findVarianceBGR(
                imgBGR, binaryImg, rows, cols, binaryPixels, blueMean,
                greenMean, redMean)

            descriptors.append(symmetryX)
            descriptors.append(symmetryY)
            descriptors.append(radius)
            descriptors.append(redMean)
            descriptors.append(redVariance)
            descriptors.append(greenMean)
            descriptors.append(greenVariance)
            descriptors.append(blueMean)
            descriptors.append(blueVariance)

            # imgBGR = mec.drawContours(imgBGR, contours, -1)
            # imgBGR = mec.drawCircle(imgBGR, (int(cX), int(cY)), 3)
            # imgBGR = mec.drawCircle(imgBGR, (int(x), int(y)), int(radius))
            # mec.showSingleImg(imgBGR, 'Modulo Extrator de Caracteristicas')
            # mec.showSingleImg(binaryImg, "Imagem Binarizada")

            benign_malignant = data.readCSV(pathImgBGR, nameImg[0])
            data.writeCSV(benign_malignant, descriptors)

            descriptors.clear()
            imgProcessed.append(nameImg[0])
            progress_bar(count, 15, 50)
        else:
            imgNotProcessed.append(nameImg[0])

    data.imageName(imgProcessed, imgNotProcessed)
    data.normalize()


def progress_bar(value, max, barsize):
    chars = int(value * barsize / float(max))
    numero = max - value
    percent = int((numero * 100) / max)
    percent = 100 - percent
    sys.stdout.write("#" * chars)
    sys.stdout.write(" " * (barsize - chars + 2))
    if value >= max:
        sys.stdout.write("Done. \n\n")
        print()
    else:
        sys.stdout.write("[%3i%%]\r" % percent)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
