import pickle
import glob

pictures = glob.glob("C:/Users/Adam/Desktop/Python/fotky/*.jpg")

#templates = glob.glob("C:/Users/Adam/Desktop/Python/Symbols/*.png")
templates_pickle = ("C:/Users/Adam/Desktop/Python/templates_dict.plk")

with open(templates_pickle, 'rb') as fd:
    templates = pickle.load(fd)

################################# vytvoření picklu pro dictionary z picklu, ktery obsahuje jen cesty k obrazkum
# templates_dict = {}
# with open("C:/Users/Adam/Desktop/Python/templates.plk", 'rb') as fd:
#     templates = pickle.load(fd)
#     for template in templates:
#         template_img = cv.imread(template)
#         templates_dict['{}'.format(template[-5])]=template_img
# with open("C:/Users/Adam/Desktop/Python/templates_dict.plk", 'wb') as _fd:
#     pickle.dump(templates_dict, _fd)

READ_SYMBOLS_SETUP = {

}

FIND_SYMBOLS_SETUP = {

}