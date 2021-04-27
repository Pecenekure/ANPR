import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt
from time import time
import config as cf
import random
import shutil

start = time()


located_chars = glob.glob("located/*.png")
templates = glob.glob("symbols2/*.png")
spz = []
overall = 0
for located_char in located_chars:
    test_img = cv.imread(located_char)
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    ret,test_img = cv.threshold(test_img,150,255,cv.THRESH_BINARY)
    print("Testovany obrazek:", located_char)
    temp = 600

    for template in templates:      
        template_img = cv.imread(template)
        template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
        ret,template_img = cv.threshold(template_img,150,255,cv.THRESH_BINARY)
        char = template[-5:-4]
        img = cv.subtract(test_img, template_img)
        reverse = cv.subtract(template_img, test_img)
        resulted = img + reverse
        eroded = cv.erode(resulted, np.ones((3,3), np.uint8), iterations=1)
        #eroded = cv.erode(eroded, (7,7), iterations=3)
        
        #eroded = cv.erode(eroded, (7,7), iterations=3)

        score = np.sum(eroded==255)
        #print("Znak {} ma score: {}".format(char,score))
        if score < temp:
            temp = score
            chosen_char = char
            # cv.imshow("sdf",resulted)
            # cv.imshow("eroze", eroded)

    if temp < 599:
        print("Zvoleny znak {} ma score: {}".format(chosen_char,temp))
        spz.append(chosen_char)
        overall += temp
        #cv.imwrite('ANPR/TrainingData/{}/Znak_pos_{0:003}.png'.format(chosen_char,random.randint(1, 99999999)), test_img)
        shutil.move(located_char, 'TrainingData/{}/{}.png'.format(chosen_char,random.randint(1, 99999999)))

    #cv.waitKey(0)
    

print("Nalezena RZ je {}".format(spz))

end = time()
print("code finished in {}s".format(end-start))

cv.imshow("image", resulted)
print("celkove skore: ", overall)

#cv.waitKey(0)