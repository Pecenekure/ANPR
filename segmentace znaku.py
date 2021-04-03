import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt
from time import time

start = time()

def rescaleFrame(frame, scale=3):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

path = glob.glob("*.png")
for i, file in enumerate(path):
    img = cv.imread(file)
    img = rescaleFrame(img)
    cv.imshow("resized {}".format(i), img)
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bilateral = cv.bilateralFilter(gray, 40, 55, 45)
    bilateral = cv.bilateralFilter(bilateral, 15, 20, 20)
    cv.imshow('bilateral', bilateral)

    ret,th1 = cv.threshold(gray,105,255,cv.THRESH_BINARY)
    cv.imshow("Thrsesholded", th1)
   

    contours, hierarchy = cv.findContours(th1, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

    for i,cnt in enumerate(contours):
        x,y,w,h = cv.boundingRect(cnt)
        print("w: {}, h:{}, xpos:{}".format(w,h,x))
        if h < 0.9*HEIGHT and h > 0.4*HEIGHT:
            if w < 0.13*WIDTH and w > 0.03*WIDTH: 
                print("znak nalezen v bode:", x)                      
                znak = th1[y:y+h,x:x+w]
                znak = cv.resize(znak,(50,100), interpolation=cv.INTER_AREA)
                cv.imshow("znak {}".format(i), znak)
                cv.imwrite('located/Znak_pos_{0:03}.png'.format(x), znak)
                print("binary image:", znak)

    th1_copy = th1.copy()
    cv.drawContours(img,contours, -1, (0,0,255), 2)
    cv.imshow("nalezene kontury", img)
    print("WIDTH: {}, HEIGHT: {}".format(WIDTH,HEIGHT))

    end = time()
    print("code finished in {}s".format(end-start))

    

    cv.waitKey(0)
    start = time()