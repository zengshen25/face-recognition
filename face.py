import numpy as np
import os
import shutil
import cv2
from PIL import Image, ImageTk
import openpyxl
import threading
import tkinter as tk


# 读取config文件，第一行代表当前已经储存的人名个数，接下来每一行是（id，name）标签和对应的人名
# 字典里存的是id——name键值对
id_dict = {}
# 已经被识别有用户名的人脸个数
Total_face_num = 999


# 将config文件内的信息读入到字典中
def init():
    f = open('./config.txt')
    global Total_face_num

    Total_face_num = len(f.readlines())
    f = open('./config.txt')
    for i in range(int(Total_face_num)):
        line = f.readline()
        id_name = line.split(' ')

        id_dict[int(id_name[0])] = id_name[1]
    f.close()

init()

# 加载OpenCV人脸检测分类器haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(r"./resources/haarcascade_frontalface_default.xml")

# 准备好识别方法LBPH方法
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 打开标号为0的摄像头
camera = cv2.VideoCapture(0)

# 从摄像头读取照片
success, img = camera.read()
W_size = 0.1 * camera.get(3)
H_size = 0.1 * camera.get(4)

 # 标志系统状态的量 0表示无子线程在运行 1表示正在刷脸 2表示正在录入新面孔。
# 相当于mutex锁，用于线程同步
system_state_lock = 0

def f_scan_face():
    pass


def f_rec_face():
    pass

def f_exit():
    pass


window = tk.Tk()
# 窗口标题
window.title('Cheney\' Face_rec 3.0')
# 这里的乘是小x
window.geometry('1000x500')
# 在图形界面上设定标签，类似于一个提示窗口的作用
var = tk.StringVar()
l = tk.Label(window, textvariable=var, bg='#dee3e9', fg='black', font=('Arial', 12), width=50, height=4)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()  # 放置l控件

# 在窗口界面设置放置Button按键并绑定处理函数
button_a = tk.Button(window, text='开始刷脸', font=('Arial', 12), width=10, height=2, command=f_scan_face)
button_a.place(x=800, y=120)

button_b = tk.Button(window, text='录入人脸', font=('Arial', 12), width=10, height=2, command=f_rec_face)
button_b.place(x=800, y=220)

button_b = tk.Button(window, text='退出', font=('Arial', 12), width=10, height=2, command=f_exit)
button_b.place(x=800, y=320)

panel = tk.Label(window, width=500, height=350)  # 摄像头模块大小
panel.place(x=10, y=100)  # 摄像头模块的位置
window.config(cursor="arrow")

# 这是我自己写的列表控件样式
listbox1 = tk.Listbox(window, borderwidth=0, activestyle='none', fg='black', background='#f0f0f0', font=18,
                      highlightthickness=7)
# 设置控件不可点击
# listbox1.config(state='disabled')
listbox1.pack(side='right', padx=228, pady=20)


def video_loop():  # 用于在label内动态展示摄像头内容（摄像头嵌入控件）
    # success, img = camera.read()  # 从摄像头读取照片
    global success
    global img
    if success:
        cv2.waitKey(1)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        window.after(1, video_loop)


video_loop()

#  窗口循环，用于显示
window.mainloop()

'''
============================================================================================
以上是关于界面的设计
============================================================================================
'''
