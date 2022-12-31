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
