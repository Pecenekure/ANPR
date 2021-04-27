import cv2 as cv
from PIL import Image
from PIL import ImageTk
import tkinter as tk

HEIGHT = 600
WIDTH = 900

root = tk.Tk()
root.title('video')

canvas = tk.Canvas(root, height=HEIGHT, width = WIDTH, bg='#d5822a')
canvas.pack()

videoFrame = tk.Frame(root)
videoFrame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.55)

videoLabel = tk.Label(videoFrame)
videoLabel.pack()



def play():
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = Image.fromarray(gray)
    gray = ImageTk.PhotoImage(gray)
    videoLabel.configure(image=gray)
    videoLabel.image = gray
    videoLabel.after(1,play)


cap = cv.VideoCapture(0)
#cap = cv.VideoCapture("videos/IMG_4726.MP4")

play()


root.mainloop()

