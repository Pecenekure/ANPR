import cv2 as cv
import numpy as np
import glob

#img = cv.imread('fotky/6.jpg')

path = glob.glob("C:/Users/Adam/Desktop/Python/fotky/*.jpg")
for file in path:
    img = cv.imread(file)

    def rescaleFrame(frame, scale=0.35):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    img = rescaleFrame(img)
    cv.imshow("resized", img)

    # Converting BGR to Grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    # Pouziti bilateralniho filtru
    #bilateral = cv.bilateralFilter(gray, 40, 55, 45)
    bilateral = cv.bilateralFilter(gray, 15, 20, 20)
    cv.imshow('bilateral', bilateral)


    ret,thresh = cv.threshold(bilateral, 100, 255, cv.THRESH_BINARY)
    cv.imshow('prahovani', thresh)

    #dilated = cv.dilate(thresh, (11,11), iterations=3)
    #cv.imshow("dilatace", dilated)

    # Nazeleni hran 
    edges = cv.Canny(gray, 170, 250)
    cv.imshow('nalezeni hran', edges)

    #dilate = cv.dilate(edges, (1,9))
    #cv.imshow("dilate", dilate)
    # Hledani krivek neboli kontur
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

    # odfiltrovani mensich kontur
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:70]
    print('pocet regionu:', len(contours))

    # vykresleni nalezenych regionu
    imgcopy = img.copy()
    cv.drawContours(imgcopy, contours, -1, (0,0,255), 2)
    cv.imshow('Nalezene regiony', imgcopy)

    cv.waitKey

    SPZ = None

    idx = 1
    for cnt in contours:
        #print("objek:", cnt)
        perimetr = cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt, 0.02 * perimetr, True)
        if len(approx) == 4:
            x,y,w,h = cv.boundingRect(cnt)
            
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
            #sirka = np.sqrt(((bottommost[0]-leftmost[0])*(bottommost[0]-leftmost[0]))+((bottommost[1]-leftmost[1])*(bottommost[1]-leftmost[1])))
            #delka = np.sqrt(((topmost[1]-leftmost[1])*(topmost[1]-leftmost[1]))+((leftmost[0]-topmost[0])*(leftmost[0]-topmost[0])))
            sirka = np.sqrt(((box[1][0]-box[0][0])*(box[1][0]-box[0][0]))+((box[1][1]-box[0][1])*(box[1][1]-box[0][1])))
            delka = np.sqrt(((box[3][1]-box[0][1])*(box[3][1]-box[0][1]))+((box[0][0]-box[3][0])*(box[0][0]-box[3][0])))
            sirka2 = np.sqrt(((box[3][0]-box[2][0])*(box[3][0]-box[2][0]))+((box[2][1]-box[3][1])*(box[2][1]-box[3][1])))
            delka2 = np.sqrt(((box[2][0]-box[1][0])*(box[2][0]-box[1][0]))+((box[2][1]-box[1][1])*(box[2][1]-box[1][1])))
            """
            print("sirka:", sirka)
            print("delka:", delka)
            print("sirka2:", sirka2)
            print("delka2:", delka2)
            print(box)
            """
            
            
            #(xx,yy),(MA,ma),angle = cv.fitEllipse(c)
            if (delka/sirka < 5.5 and delka/sirka > 3.5) or (sirka/delka < 5.5 and sirka/delka > 3.5):
                if sirka > 0.85*sirka2 and sirka <1.05*sirka2 and delka > 0.95*delka2 and delka < 1.05*delka2:
                    SPZ = approx 
                    cv.drawContours(img, [SPZ], -1, (0,0,255), 2) 
                    new_img = img[y:y+h,x:x+w]
                    cv.imwrite('Oriznuty obrazek ' + str(idx) + '.png', new_img)
                    #cv.drawContours(img,[box],0,(0,255,0),5)
                    idx+=1
                #break
            else:
                cv.drawContours(img, [cnt], -1, (255,0,255), 2)
                new_img = img[y:y+h,x:x+w]
                cv.imwrite('Oriznuty obrazek ' + str(idx) + '.png', new_img)
                idx+=1


    #cv.drawContours(img, [SPZ], -1, (0,0,255), 2)
    cv.imshow('Nalezena SPZ', img)

    #cv.imshow('Oriznuty obrazek:', cv.imread('Oriznuty obrazek 1.png'))


    cv.waitKey(0)
