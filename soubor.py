import cv2 as cv
import numpy as np

img = cv.imread('fotky/auto.jpg')

def rescaleFrame(frame, scale=0.35):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_img = rescaleFrame(img)
cv.imshow("resized", resized_img)

blank = np.zeros(resized_img.shape[:2], dtype='uint8')
b,g,r = cv.split(resized_img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('Blue',blue)
cv.imshow('Green',green)
cv.imshow('Red',red)
 
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

cv.waitKey(0)
