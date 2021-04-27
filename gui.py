import tkinter as tk
import get_video
import config as cf
import cv2 as cv
import os
from PIL import Image
from PIL import ImageTk
import compare

HEIGHT = 600
WIDTH = 900
method = 'cnn'

#source = "videos/IMG_4726.MP4"
source = 0
cap = cv.VideoCapture(source)



def play():
    ret, frame = cap.read()
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray = gray[200:500, 100:1100]
    # img = frame[200:500, 100:1100]
    useless_number, located_frame = compare.performRecognition(frame, method=method)     ###### TADY VOLAM FUNKCIONALITU
    update()
    RGB = cv.cvtColor(located_frame, cv.COLOR_BGR2RGB)
    RGB = Image.fromarray(RGB)
    RGB = ImageTk.PhotoImage(RGB)
    videolabel.configure(image=RGB)
    videolabel.image = RGB
    videolabel.after(1,play)



def CNNButtonClicked():
    method = 'cnn'



def compareFunctionButtonClicked():
    method = 'compare'


def update():
    pass


###############################################################################################################
root = tk.Tk()
root.geometry("800x600")
canvas = tk.Canvas(root)#, relheight=HEIGHT, width = WIDTH, bg='#d5822a')
canvas.pack()

##########################################################      VIDEO A SPZ FRAME
videoFrame = tk.Frame(root, bg='#3399ff')
videoFrame.place(relx=0.05, rely=0.05, relwidth=0.6, relheight=0.55)

videolabel = tk.Label(videoFrame)
videolabel.place(relwidth=1, relheight=1)


spzFrame = tk.Frame(root, bg='#3399ff')
spzFrame.place(relx=0.05, rely=0.65, width=320, height=100)

spz_img = tk.PhotoImage(file='gui_pics/Oriznuty obrazek.png')
spzImgLabel = tk.Label(spzFrame, image=spz_img)
spzImgLabel.place(relwidth=1, relheight=1)

#######################################################################################  METHOD SELECTION
buttonsFrame = tk.LabelFrame(root, text = 'setup', bg='#3399ff', bd = 5)
buttonsFrame.place(anchor='ne', relx=0.95, rely=0.05, relwidth=0.25, relheight=0.16)

CNNbutton = tk.Button(buttonsFrame, text='CNN', padx = 20, pady = 20, command = lambda: CNNButtonClicked())
CNNbutton.place(relx=0.025, rely=0.5, relwidth=0.45, relheight=0.45)

compareFunctionButton = tk.Button(buttonsFrame, text='Compare', padx = 20, pady = 20, command = lambda: compareFunctionButtonClicked())
compareFunctionButton.place(relx=0.525, rely=0.5, relwidth=0.45, relheight=0.45)

OCRlabel = tk.Label(buttonsFrame, text='Select OCR algorithm', bg='#3399ff', font =40)
OCRlabel.place(anchor='n', relx=0.5, relheight='0.5')

########################################################################################## RESULT FRAME
resultFrame = tk.Frame(root, bg='#3399ff', bd = 5)
resultFrame.place(relx=0.5, rely=0.65, relwidth=0.4, relheight=0.2)

label = tk.Label(resultFrame, text='Nalezena SPZ',bd = 10, bg='#3399ff', font=40)
label.place(anchor='n', relx=0.2, relheight='0.5')

spzLabel = tk.Label(resultFrame, text='spz')
spzLabel.place(rely = 0.5, relwidth=1, relheight=0.5)




play()
root.mainloop()