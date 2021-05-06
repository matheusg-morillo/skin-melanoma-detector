from MEC import Features_extractor
from MPP import Pre_Processing_Module
from Data import Data_Extrator
import os
import sys


def main():
    mpp = Pre_Processing_Module.PreProcessing_Module()
    mec = Features_extractor.Features_Extractor()
    data = Data_Extrator.Data_Extrator(r'C:\Users\rapha\Documents\Images_BMP\\')

    pathSegmentation = r'C:\Users\rapha\Documents\Images_Seg_BMP'
    pathImgBGR = r'C:\Users\rapha\Documents\Images_BMP'
    descriptors = []
    index = 0
    count = 0

    for arquivo in os.listdir(
            pathSegmentation):
        count += 1
        nameImg = arquivo.split('.')
        binaryImg = mpp.openImg(pathSegmentation + '\\' + nameImg[0] + '.bmp')
        imgBGR = mpp.openImg(pathImgBGR + '\\' + nameImg[0] + '.bmp')
        binaryImg = mpp.resizeImg(binaryImg, (600, 400))
        imgBGR = mpp.resizeImg(imgBGR, (600, 400))
        # grayImg = mpp.bgr2Gray(binaryImg)
        # grayImg = mpp.equalizeImg(grayImg)
        # gaussianBlurImg = mpp.gaussianBlur(grayImg, blur)
        # (T, binaryImg) = mpp.binaryImg(grayImg, 120, 255)
        # binaryImg = mpp.morphologyClose(binaryImg)
        # borderedImg = mpp.canny(binaryImg)
        contours, hierarchy = mec.findContours(binaryImg)

        moments = mec.findMoments(binaryImg)
        (cX, cY) = mec.findCenterOfMass(moments)
        (rows, cols) = binaryImg.shape
        binaryPixels = mec.findBinaryPixels(binaryImg, rows, cols)
        symmetryX = mec.findSymmetryX(binaryImg, rows, cols, cX)
        symmetryY = mec.findSymmetryY(binaryImg, rows, cols, cY)
        (x, y), diameter = mec.findMinEnclosingCircle(contours[index])
        (blueMean, greenMean, redMean) = mec.findMeanBGR(imgBGR, binaryImg,
                                                         rows,
                                                         cols,
                                                         binaryPixels)
        (blueVariance, greenVariance, redVariance) = mec.findVarianceBGR(
            imgBGR, binaryImg, rows, cols, binaryPixels, blueMean,
            greenMean, redMean)

        descriptors.append(symmetryX)
        descriptors.append(symmetryY)
        descriptors.append(diameter)
        descriptors.append(redMean)
        descriptors.append(redVariance)
        descriptors.append(greenMean)
        descriptors.append(greenVariance)
        descriptors.append(blueMean)
        descriptors.append(blueVariance)

        # imgBGR = mec.drawContours(imgBGR, contours, -1)
        # imgBGR = mec.drawCircle(imgBGR, (int(cX), int(cY)), 3)
        # imgBGR = mec.drawCircle(imgBGR, (int(x), int(y)), int(diameter))
        # mec.showSingleImg(imgBGR, 'Modulo Extrator de Caracteristicas')
        # mec.showSingleImg(binaryImg, "Imagem Binarizada")

        benign_malignant = data.readCSV(r'C:\Users\rapha\Documents\ISIC-download', nameImg[0])
        data.writeCSV(benign_malignant, descriptors)

        descriptors.clear()
        progress_bar(count, len(os.listdir(pathSegmentation)), 50)

    data.normalize()


def normalize():
    data = Data_Extrator.Data_Extrator(
        r'Caminho dos dados processados')
    data.normalize()


def convert():
    mpp = Pre_Processing_Module.PreProcessing_Module()
    count = 0
    pathImg = r'C:\Users\rapha\Documents\ISIIC_Download_Segmentation\\'
    pathImgBMP = r'C:\Users\rapha\Documents\Images_Seg_BMP\\'

    for arquivo in os.listdir(r'C:\Users\rapha\Documents\Images_JPG\\'):
        count += 1
        nameImg = arquivo.split('.')
        mpp.convertToBMP(pathImg + nameImg[0] + '.png', pathImgBMP + nameImg[0] + '.bmp')
        progress_bar(count, len(os.listdir(pathImg)), 50)


def histograma():
    mpp = Pre_Processing_Module.PreProcessing_Module()
    mec = Features_extractor.Features_Extractor()
    grayImg = mpp.bgr2Gray(mpp.openImg(r'Caminho da Imagem'))
    # grayImg = mpp.resizeImg(grayImg, (400, 400))
    mec.showSingleImg(grayImg, "Gray Image")
    histograma = mpp.histograma(grayImg)
    mpp.plotHist(histograma)


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
    # convert()
    # normalize()
    # histograma()
