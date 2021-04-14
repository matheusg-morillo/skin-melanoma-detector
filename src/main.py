from MEC import Features_extractor
from MPP import Pre_Processing_Module
from Data import Data_Extrator
import os


def main():
    mpp = Pre_Processing_Module.PreProcessing_Module()
    mec = Features_extractor.Features_Extractor()
    data = Data_Extrator.Data_Extrator(
        r'C:\Users\rapha\Documents\skin-melanoma-detector\src\Data\\')

    pathSegmentation = r'C:\Users\rapha\Documents\ISIIC_Download_Segmentation'
    pathImgBGR = r'C:\Users\rapha\Documents\ISIC-download'
    descriptors = []
    index = 0

    for arquivo in os.listdir(
            r'C:\Users\rapha\Documents\ISIIC_Download_Segmentation'):
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
        moments = mec.findMoments(binaryImg)
        (cX, cY) = mec.findCenterOfMass(moments)
        (rows, cols) = binaryImg.shape
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
        print('Size: %s' % (len(contours)))

        # imgBGR = mec.drawContours(imgBGR, contours, -1)
        # imgBGR = mec.drawCircle(imgBGR, (int(cX), int(cY)), 3)
        # imgBGR = mec.drawCircle(imgBGR, (int(x), int(y)), int(radius))
        # mec.showSingleImg(imgBGR, 'Modulo Extrator de Caracteristicas')
        # mec.showSingleImg(binaryImg, "Imagem Binarizada")

        benign_malignant = data.readCSV(pathImgBGR, nameImg[0])
        data.writeCSV(str(nameImg[0]), benign_malignant, descriptors)

        descriptors.clear()


if __name__ == "__main__":
    main()
