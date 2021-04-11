from MEC import Features_extractor
from MPP import Pre_Processing_Module


def main():
    mpp = Pre_Processing_Module.PreProcessing_Module(
        r'C:\Users\rapha\Documents\ISIC-download\ISIC_0000001.jpg')
    mec = Features_extractor.Features_Extractor()

    descriptors = []

    imgMask = imgBGR = mpp.openImg()
    grayImg = mpp.bgr2Gray(imgMask)
    gaussianBlurImg = mpp.gaussianBlur(grayImg, (7, 7))
    (T, binaryImg) = mpp.binaryImg(gaussianBlurImg, 120, 255)
    binaryImg = mpp.morphologyClose(binaryImg)
    borderedImg = mpp.canny(binaryImg)
    contours, hierarchy = mec.findContours(borderedImg)
    moments = mec.findMoments(borderedImg)
    (cX, cY) = mec.findCenterOfMass(moments)
    (rows, cols) = grayImg.shape
    binaryPixels = mec.findBinaryPixels(binaryImg, rows, cols)
    symmetryX = mec.findSymmetryX(binaryImg, rows, cols, cX)
    symmetryY = mec.findSymmetryY(binaryImg, rows, cols, cY)
    (x, y), radius = mec.findMinEnclosingCircle(contours[0])
    (blueMean, greenMean, redMean) = mec.findMeanBGR(imgBGR, binaryImg,
                                                     rows, cols, binaryPixels)
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


if __name__ == "__main__":
    main()
