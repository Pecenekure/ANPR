import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt
from time import time


spz = []
templates = glob.glob("C:/Users/Adam/Desktop/Python/Symbols/*.png")

test_img = cv.imread("C:/Users/Adam/Desktop/Python/located/Znak_pos_401.png")
test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
ret,test_img = cv.threshold(test_img,150,255,cv.THRESH_BINARY)

for template in templates:      
    template_img = cv.imread(template)
    template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    ret,template_img = cv.threshold(template_img,150,255,cv.THRESH_BINARY)
    char = template[-5:-4]
    img = cv.subtract(test_img, template_img)
    reverse = cv.subtract(template_img, test_img)
    resulted = img + reverse
    eroded = cv.erode(resulted, (7,7), iterations=3)
    """
    eroded = cv.erode(eroded, (7,7), iterations=3)

    eroded = cv.erode(eroded, (7,7), iterations=3) """
    cv.imshow("eroded", eroded)

    score = np.sum(eroded==255)
    print(score)
    print("Znak {} ma score: {}".format(char,score))

    cv.waitKey(0)