import cv2 as cv
import numpy as np

img = cv.imread('fotky/3.jpg')

def rescaleFrame(frame, scale=0.35):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleFrame(img)
img = img[100:450, 50:390]
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("resized", img)

#Averaging
average = cv.blur(img, (5,5))
cv.imshow('Average blur', average)

median = cv.medianBlur(img, 5)
cv.imshow('Median', median)

gauss = cv.GaussianBlur(img, (5,5), 0)
cv.imshow('Gaussian', gauss)

bil = cv.bilateralFilter(img, 40, 55, 45)
cv.imshow("bilateral", bil)

bil = cv.bilateralFilter(bil, 15,20,20)
cv.imshow('bilateral2', bil)


cv.waitKey(0)
