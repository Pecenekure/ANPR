import tkinter as tk
import get_video
import config as cf
import cv2 as cv
import os
from PIL import Image
from PIL import ImageTk
import compare





class Play:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent)  # , relheight=HEIGHT, width = WIDTH, bg='#d5822a')
        self.canvas.pack()
        ################################################################## VSTUPNI PARAMETRY
        self.source = 0
        #self.source = "videos/IMG_4726.MP4"
        self.license_plate = {'spz':'nic', 'car':'nic', 'distance':0}
        self.method = 'cnn'
        self.cap = cv.VideoCapture(self.source)
        ##########################################################      VIDEO A SPZ PICTURE FRAME INIT
        self.videoFrame = tk.Frame(parent, bg='#3399ff')
        self.videoFrame.place(relx=0.05, rely=0.05, relwidth=0.6, relheight=0.55)

        self.videolabel = tk.Label(self.videoFrame)
        self.videolabel.place(relwidth=1, relheight=1)

        self.spzFrame = tk.Frame(parent, bg='#3399ff')
        self.spzFrame.place(relx=0.05, rely=0.65, width=320, height=100)

        #######################################################################################  METHOD SELECTION FRAME INIT
        self.buttonsFrame = tk.LabelFrame(parent, text='setup', bg='#3399ff', bd=5)
        self.buttonsFrame.place(anchor='ne', relx=0.95, rely=0.05, relwidth=0.25, relheight=0.16)

        self.CNNbutton = tk.Button(self.buttonsFrame, text='CNN', padx=20, pady=20, command= self.CNNButtonClicked)
        self.CNNbutton.place(relx=0.025, rely=0.5, relwidth=0.45, relheight=0.45)

        self.compareFunctionButton = tk.Button(self.buttonsFrame, text='Compare', padx=20, pady=20, command= self.compareFunctionButtonClicked)
        self.compareFunctionButton.place(relx=0.525, rely=0.5, relwidth=0.45, relheight=0.45)

        self.OCRlabel = tk.Label(self.buttonsFrame, text='Select OCR algorithm', bg='#3399ff', font=40)
        self.OCRlabel.place(anchor='n', relx=0.5, relheight='0.5')

        self.methodStatusLabel = tk.Label(parent, text = self.method + ' method runing...')
        self.methodStatusLabel.place(anchor = 'ne', relx = 0.95, rely = 0.21)

        ########################################################################################## RESULT FRAME INIT
        self.resultFrame = tk.Frame(parent, bg='#3399ff', bd=5)
        self.resultFrame.place(relx=0.5, rely=0.65, relwidth=0.4, relheight=0.2)

        self.label = tk.Label(self.resultFrame, text='Nalezena SPZ', bd=10, bg='#3399ff', font=40)
        self.label.place(anchor='n', relx=0.2, relheight='0.5')

        self.spzLabel = tk.Label(self.resultFrame, text='spz', font = 120)
        self.spzLabel.place(rely=0.5, relwidth=1, relheight=0.5)

        self.authenticationLabel = tk.Label(parent, text='cgnjhfd', font=60)
        self.authenticationLabel.place(relx = 0.5, rely=0.85, relwidth=0.4, relheight=0.1)

        self.distanceLabel = tk.Label(parent, text='cgnjhfd', font=60)
        self.distanceLabel.place(anchor='ne', relx=0.95, rely=0.30, relwidth=0.25, relheight=0.16)

        self.refresh()

    def CNNButtonClicked(self):
        self.method = 'cnn'

    def compareFunctionButtonClicked(self):
        self.method = 'compare'

    def refresh(self):
        """ refresh the content of the label every 10 milisecond """
        if len(self.license_plate['spz']) == 7:
            self.spzLabel.configure(text=self.license_plate['spz'], font = 120)
            self.authenticationLabel.configure(text="Registered car: {}".format(self.license_plate['car'][0]), font = 60)

        self.methodStatusLabel.configure(text = self.method + ' method runing...')


        ret, frame = self.cap.read()
        succesful, located_frame, self.license_plate = compare.performRecognition(frame, method= self.method)  ###### TADY VOLAM FUNKCIONALITU
        if succesful:
            self.spz_img = tk.PhotoImage(file='gui_pics/Oriznuty obrazek.png')
            self.spzImgLabel = tk.Label(self.spzFrame, image=self.spz_img)
            self.spzImgLabel.place(relwidth=1, relheight=1)
            self.distanceLabel.configure(text='Distance: {}'.format(str(self.license_plate['distance'])))
        RGB = cv.cvtColor(located_frame, cv.COLOR_BGR2RGB)
        RGB = Image.fromarray(RGB)
        RGB = ImageTk.PhotoImage(RGB)
        self.videolabel.configure(image=RGB)
        self.videolabel.image = RGB
        self.videolabel.after(1, self.refresh)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    timer = Play(root)
    root.mainloop()