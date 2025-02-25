import tkinter as tk
win = tk.Tk()
win.geometry("500x500")
win.title("Title")

btn1 = tk.Button(win, width = 25, text = 'First Button')
btn1.pack()
btn2 = tk.Button(win, width = 25, text = 'Second Button')
btn2.pack()

win.mainloop()