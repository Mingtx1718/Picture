import cv2
import requests
import json
import base64
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import (askopenfilename, asksaveasfilename)
from PIL import Image, ImageTk
import os
import threading
import pyaudio
import wave
import speech_recognition as sr
from win32com.client import Dispatch
import pyttsx3

# opencv图像转base64
def image2base64(image_np, format):
    image = cv2.imencode(format, image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


# base64转opencv图像
def base642image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


# 向face++进行请求
def link2facepp(img1, format1, img2, format2):
    img64_1 = image2base64(img1, format1)
    img64_2 = image2base64(img2, format2)
    url = 'https://api-cn.faceplusplus.com/imagepp/v1/mergeface'
    data = {
        "api_key": "i9ac-ZZMEOWqIlRIiLD3oiR26OpAWM43",
        "api_secret": "wad4LNBj41EiWxreH204QlN6WRaOLUZy",
        "template_base64": img64_1,
        "merge_base64": img64_2,
        "merge_rate": 50
    }
    response1 = requests.post(url, data=data)
    return response1


# 解码face++返回的图像为opencv的格式
def ReadFromResponse(response):
    res_con = response.content.decode('utf-8')
    res_dict = json.JSONDecoder().decode(res_con)
    result = res_dict['result']
    img = base642image(result)
    return img


# 构建手写数字识别的knn
def Tr(knn):
    img = cv2.imread('d.png', cv2.IMREAD_GRAYSCALE)
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
    x = np.array(cells)
    train = x[:, :].reshape(-1, 400).astype(np.float32)
    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    return knn


# 使用knn进行手写数字识别
def rec(knn, gray):
    gray = cv2.resize(gray, (20, 20))
    test = gray[:, :].reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    result = np.array(result)
    return result[0, 0].astype(np.uint8)


# 模型集合
model = ['candy.t7', 'composition_vii.t7', 'feathers.t7', 'la_muse.t7', 'mosaic.t7', 'starry_night.t7', 'the_scream.t7', 'the_wave.t7', 'udnie.t7']


def Style_Transfer(net, img):
    img = cv2.resize(img, (400, 400))
    img = cv2.GaussianBlur(img, (7, 7), 0, 0)
    Mean = (np.mean(img[0]), np.mean(img[1]), np.mean(img[2]))
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.1, (w, h), Mean, swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += Mean[0]
    out[1] += Mean[1]
    out[2] += Mean[2]
    out = out.transpose(1, 2, 0)
    cv2.imwrite('temp.png', out)
    out = cv2.imread('temp.png')
    os.remove('temp.png')
    out = cv2.resize(out, (400, 400))
    mix = 0.5
    M = cv2.medianBlur(img, 5)
    out = cv2.addWeighted(out, mix, M, 1 - mix, 0)
    return out


class Recorder:
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.file = "test.wav"

    def start(self):
        threading._start_new_thread(self._recording, ())

    def _recording(self):
        self._running = True
        print('start recording')
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while self._running:
            data = stream.read(self.CHUNK)
            self._frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def save(self):
        p = pyaudio.PyAudio()

        wf = wave.open(self.savePath(), 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()

    def savePath(self):
        path_ = asksaveasfilename(title='保存一个.wav文件', initialdir='/',
                                  filetypes=[('Wave File', '*.wav')])  # 使用askdirectory()方法返回文件夹的路径
        if path_ == "":
            self.savePath(self)  # 当打开文件路径选择框后点击"取消" 输入框会清空路径，所以使用get()方法再获取一次路径
        else:
            path_ = path_.replace("/", "\\")  # 实际在代码中执行的路径为“\“ 所以替换一下
            return path_

    def stop(self):
        self._running = False
        print('stop')

    def play_audio(self, filename=""):
        CHUNK = 1024
        if filename == "":
            filename = self.file
        wf = wave.open(filename, 'rb')
        data = wf.readframes(CHUNK)
        p = pyaudio.PyAudio()

        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()

        # print('FORMAT: {} CHANNELS: {} RATE: {}'.format(FORMAT, CHANNELS, RATE))

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        frames_per_buffer=CHUNK,
                        output=True)

        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(CHUNK)

    @staticmethod
    def rec_audio(file):
        """
        :param: The Absolute Path of the audio file
        :return: Content identified
        """
        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            audio = r.record(source)
        try:
            # print(r.recognize_sphinx(audio, language='zh-CN'))
            # print(r.recognize_google(audio, language='zh-CN'))
            return r.recognize_google(audio, language='zh-CN')
        except Exception as e:
            print(e)