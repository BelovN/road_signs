import tkinter as tk
from tkinter import *
from tkinter import filedialog
from detecter import predict
from PIL import ImageTk
import os

ROOT_DIR = os.path.abspath(os.curdir)


def browse_file():
    filename = filedialog.askopenfilename()

    predict(filename)
    img  =  ImageTk.PhotoImage(file=ROOT_DIR + "\\output\\sign_output.png")
    canvas = Canvas(root, width=img.width(), height=img.height())
    canvas.pack()
    canvas.create_image(img.width()/2, img.height()/2, image=img)
    root.mainloop()

root = Tk()


def main():
    menu = Menu(root)
    root.config(menu=menu)
    menu.add_cascade(label="Open", command=browse_file)
    root.mainloop()

if __name__ == "__main__":
    main()
