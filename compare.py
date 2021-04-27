import cv2 as cv
import numpy as np
import pandas as pd
import shutil
from matplotlib import pyplot as plt
from time import time
import config as cf
import cProfile, pstats, io
import random



pr = cProfile.Profile()
pr.enable()

path = cf.pictures
database = pd.read_excel('database.xlsx')


def rescaleFrame(frame, scale=3): #################################################################################################
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def readSymbols(symbols, method = 'cnn'): #########################################################################################################
    located_chars = symbols #glob.glob("C:/Users/Adam/Desktop/Python/located/*.png")
    #templates = glob.glob("C:/Users/Adam/Desktop/Python/Symbols/*.png")
    templates = cf.templates
    spz = []
    probabiliy = []
    #print("symbols:", symbols.keys())

    if method == 'cnn':
        for keys, value in sorted(located_chars.items()):
            test_img = value
            test_img = np.asarray(test_img)
            test_img = cv.resize(test_img,(25,50))
            test_img = test_img/255
            test_img = test_img.reshape(1,50,25,1)

            chosen_char_index = int(cf.model.predict_classes(test_img))
            chosen_char = cf.myList[chosen_char_index]
            predictions = cf.model.predict(test_img)
            prob_val = np.amax(predictions)

            if prob_val > 0.9:
                spz.append(chosen_char)
                probabiliy.append(prob_val)

    elif method == 'compare':
        for keys, value in sorted(located_chars.items()):
            test_img = value
            #test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
            ret,test_img = cv.threshold(test_img,140,255,cv.THRESH_BINARY)
            temp = 1000
            for template in templates:
                #template_img = cv.imread(template)
                template_img = templates[template]
                template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
                ret,template_img = cv.threshold(template_img,150,255,cv.THRESH_BINARY)
                char = template
                img = cv.subtract(test_img, template_img)
                reverse = cv.subtract(template_img, test_img)
                resulted = img + reverse
                #eroded = cv.erode(resulted, (7,7), iterations=3)
                eroded = cv.erode(resulted, np.ones((3,3), np.uint8), iterations=1)
                #eroded = cv.erode(eroded, (7,7), iterations=3)

                score = np.sum(eroded==255)
                if score < temp:
                    temp = score
                    chosen_char = char
                #print("Znak {} ma score: {}".format(char,score))

            if temp < 1000:
                #print("Zvoleny znak {} ma score: {}".format(chosen_char,temp))
                spz.append(chosen_char)

    return spz, probabiliy

def findSymbols(img, threshold): #############################################################################################################
    img = rescaleFrame(img,3)
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    symbols = {}

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #bilateral = cv.bilateralFilter(gray, 40, 55, 45)
    #bilateral = cv.bilateralFilter(bilateral, 15, 20, 20)
    ret,th1 = cv.threshold(gray,threshold,255,cv.THRESH_BINARY)
    #cv.imshow("threshold", th1)
    contours, hierarchy = cv.findContours(th1, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

    for i,cnt in enumerate(contours):
        x,y,w,h = cv.boundingRect(cnt)
        if h < 0.9*HEIGHT and h > 0.4*HEIGHT:
            if w < 0.13*WIDTH and w > 0.025*WIDTH:
                znak = th1[y:y+h,x:x+w]
                znak = cv.resize(znak,(50,100), interpolation=cv.INTER_AREA)
                symbols['znak_pos_{0:03}'.format(x)] = znak
                #cv.imwrite('located/Znak_pos_{0:003}.png'.format(random.randint(1,99999999)), znak)

    return symbols

def performRecognition(img, method = 'cnn'):  ####################################################################################################################
    #print("\n PROCESS ZAVOLAN")
    #print('{} method runing....'.format(method))
    start = time()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img[200:500, 100:1100]
    gray = gray[200:500, 100:1100]
    img_copy = img.copy()
    cislo = 0
    succesfull_recognitions = 0

    # Pouziti bilateralniho filtru
    # bilateral = cv.bilateralFilter(gray, 40, 55, 45)
    # bilateral = cv.bilateralFilter(bilateral, 15, 20, 20)
    bilateral2 = cv.bilateralFilter(gray, 11, 17, 17)

    # Nazeleni hran
    edges = cv.Canny(gray, 170, 250)
    edges2 = cv.Canny(gray, 100, 250)
    edges3 = cv.Canny(bilateral2, 100, 250)

    # Hledani kontur
    # contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    # contours2, hierarchy2 = cv.findContours(edges2, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    # contours3, hierarchy3 = cv.findContours(edges3, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    # contours = contours3 + contours2 + contours

    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv.findContours(edges2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv.findContours(edges3, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = contours3 + contours2 + contours

    # odfiltrovani mensich kontur pro urychleni kodu
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:50]

    # vykresleni nalezenych regionu
    imgcopy = img.copy()
    cv.drawContours(imgcopy, contours, -1, (0, 0, 255), 2)
    #cv.imshow('Contours', imgcopy)

    SPZ_conture = None
    idx = 1
    authenticated_spz =  {'spz':'nic', 'car':'nic', 'distance':0}
    acces = False
    for cnt in contours:
        if acces == False:
            # print("object:", cnt)
            perimetr = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.03 * perimetr, True)
            if len(approx):  # ==4 pokud chci kontrolovat ze kontura je obdelnik
                x, y, w, h = cv.boundingRect(cnt)

                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)

                sirka = np.sqrt(((box[1][0] - box[0][0]) * (box[1][0] - box[0][0])) + (
                            (box[1][1] - box[0][1]) * (box[1][1] - box[0][1])))
                delka = np.sqrt(((box[3][1] - box[0][1]) * (box[3][1] - box[0][1])) + (
                            (box[0][0] - box[3][0]) * (box[0][0] - box[3][0])))
                sirka2 = np.sqrt(((box[3][0] - box[2][0]) * (box[3][0] - box[2][0])) + (
                            (box[2][1] - box[3][1]) * (box[2][1] - box[3][1])))
                delka2 = np.sqrt(((box[2][0] - box[1][0]) * (box[2][0] - box[1][0])) + (
                            (box[2][1] - box[1][1]) * (box[2][1] - box[1][1])))

                if delka!=0 and sirka!=0 and ((delka / sirka < 5.5 and delka / sirka > 3.5) or (sirka / delka < 5.5 and sirka / delka > 3.5)):
                    if sirka > 0.85 * sirka2 and sirka < 1.05 * sirka2 and delka > 0.95 * delka2 and delka < 1.05 * delka2:
                        area = cv.contourArea(cnt)
                        if sirka * delka > 0.85 * area and sirka * delka < 1.15 * area:

                            ################# FUNKCE NA MERENI VZDALENOSTI
                            authenticated_spz['distance'] = 1/sirka
                            #print("Vzdalenost: ", sirka, sirka2, delka, delka2)
                            ###############
                            SPZ_conture = cnt
                            idx += 1
                            spz_Image = img_copy[y:y + h, x:x + w]
                            cv.drawContours(img, [SPZ_conture], -1, (0, 0, 255), 2)
                            img_for_gui = cv.resize(spz_Image, (320,100), interpolation=cv.INTER_AREA)
                            #cv.imwrite('Oriznuty obrazek ' + str(idx) + '.png', spz_Image)
                            symbols = findSymbols(spz_Image, 150)
                            _symbols = findSymbols(spz_Image, 100)
                            spz, prob = readSymbols(symbols, method)
                            _spz, _prob = readSymbols(_symbols, method)
                            if len(spz) == 7 or len(_spz) == 7:
                                acces = True
                                cv.imwrite('gui_pics/Oriznuty obrazek.png', img_for_gui)
                                print("Nalezene SPZ: ", spz, _spz, "Probabilities:", prob, _prob)
                                spz = ''.join(map(str, spz))
                                _spz = ''.join(map(str, _spz))
                                if (database.SPZ == spz).any():
                                    succesfull_recognitions += 1
                                    index = database.loc[database.SPZ == spz].index
                                    print("Registered car: {}, Access granted!".format(
                                        database.loc[database.SPZ == spz].Names.values))
                                    print(len(symbols))
                                    #cv.imwrite('Obrazek ' + str(counter) + '.png', img_copy)
                                    authenticated_spz['spz'] = spz
                                    authenticated_spz['car'] = database.loc[database.SPZ == spz].Names.values

                                elif (database.SPZ == _spz).any():
                                    succesfull_recognitions += 1
                                    index = database.loc[database.SPZ == _spz].index
                                    print("Registered car: {}, Access granted!".format(
                                        database.loc[database.SPZ == _spz].Names.values))
                                    print(len(symbols))
                                    #cv.imwrite('Obrazek ' + str(counter) + '.png', img_copy)
                                    #authenticated_spz = {'spz':_spz, 'car':database.loc[database.SPZ == _spz].Names.values}
                                    authenticated_spz['spz'] = _spz
                                    authenticated_spz['car'] = database.loc[database.SPZ == _spz].Names.values

                        spz = None
                        _spz = None



        else:
            break

    #print("pocet ulozenych obrazku:", idx)
    cv.imshow('Nalezena SPZ', img)

    #cv.imshow('Oriznuty obrazek:', cv.imread('Oriznuty obrazek 1.png'))

    cislo += 1
    #print("Testovany obrzaek c.: ", file[-6:])

    end = time()
    #print("\ncode finished in {}s\n".format(end-start))

    return succesfull_recognitions, img, authenticated_spz



if __name__ == "__main__":
    total_tested = 0
    succesfull_recognitions = 0
    for file in sorted(path):
        img = cv.imread(file)
        #img = rescaleFrame(img, scale=0.4)
        total_tested += 1

        # Converting BGR to Grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        start = time()
        success, result_img = performRecognition(gray, img)
        succesfull_recognitions += success

        #cv.imshow('Nalezena SPZ', result_img)
        #cv.waitKey(0)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
    print("uspesne rozpoznanych znacek: {} z {} testovanych obrazku".format(succesfull_recognitions, total_tested))
    