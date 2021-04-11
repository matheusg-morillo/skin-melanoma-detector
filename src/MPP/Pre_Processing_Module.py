import cv2


class PreProcessing_Module(object):
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
