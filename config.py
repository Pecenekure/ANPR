import pickle
import glob
import cv2 as cv
import os

#method = 'cnn'
method = 'compare'



pictures = glob.glob("fotky/*.jpg")
templates_pickle = ("templates_dict.plk")
model_pickle = ("trained_model_2konvoluce.plk")
myList = os.listdir('TrainingData')

with open(model_pickle, "rb") as pickleIn:
    model = pickle.load(pickleIn)

try:
    with open(templates_pickle, 'rb') as fd:
        templates = pickle.load(fd)
except: ################################# vytvoření picklu s templaty pokud neni
    templates_dict = {}
    templates = glob.glob("symbols/*.png")
    for template in templates:
        template_img = cv.imread(template)
        templates_dict['{}'.format(template[-5])]=template_img
    with open("templates_dict.plk", 'wb') as _fd:
        pickle.dump(templates_dict, _fd)
        templates = templates_dict





READ_SYMBOLS_SETUP = {

}

FIND_SYMBOLS_SETUP = {

}