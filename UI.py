from lib import *


class Window:
    def __init__(self):
        self.win = Tk()

    def SetWindow(self):
        print('')

    def OpenWindow(self):
        self.win.mainloop()


class Window1(Window):
    '''
    换脸窗口
    '''

    # 初始化函数
    def __init__(self, par):
        super().__init__()
        self.par = par
        self.img1 = np.zeros((400, 400, 3), np.uint8)
        self.img1.fill(255)
        self.img2 = np.zeros((400, 400, 3), np.uint8)
        self.img2.fill(255)
        self.imgf1 = None
        self.imgf2 = None
        self.p1 = None
        self.p2 = None
        # print('这是初始化函数')

    # 将Label p 的内容设置为 图片img 的内容
    def setpannle(self, img, p):
        current_image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=current_image)
        p.imgtk = imgtk
        p.config(image=imgtk)

    # 从文件选择图片1
    def Getimg1(self):
        path = askopenfilename(title='窗口标题', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        path.replace("/", "\\")
        self.img1 = cv2.imread(path)
        self.img1 = cv2.resize(self.img1, (400, 400))
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.imgf1 = path[len(path) - 4:len(path)]
        self.setpannle(self.img1, self.p1)
        # print('从文件选择图片1')

    def Getimg2(self):
        path = askopenfilename(title='窗口标题', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        path.replace("/", "\\")
        self.img2 = cv2.imread(path)
        self.img2 = cv2.resize(self.img2, (400, 400))
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        self.imgf2 = path[len(path) - 4:len(path)]
        self.setpannle(self.img2, self.p2)
        # print('从文件选择图片2')

    def FaceMerge(self):
        response = link2facepp(self.img1, self.imgf1, self.img2, self.imgf2)
        self.img1 = ReadFromResponse(response)
        self.setpannle(self.img1, self.p1)
        # print('换脸')

    def Main(self):
        self.win.destroy()
        self.par.__init__()
        self.par.SetWindow()
        self.par.OpenWindow()

    def SetWindow(self):
        self.win.title("换脸")
        self.win.geometry("800x500")
        f1 = Frame(self.win)
        f2 = Frame(self.win)
        f3 = Frame(self.win)
        f1.pack(side=TOP)
        f2.pack(side=TOP)
        f3.pack(side=BOTTOM)
        Button(f1, text='原图', command=self.Getimg1).pack(side=LEFT)
        Button(f1, text='脸图', command=self.Getimg2).pack(side=LEFT)
        Button(f1, text='换脸', command=self.FaceMerge).pack(side=LEFT)
        self.p1 = Label(f2)
        self.p1.pack(side=LEFT)
        self.p2 = Label(f2)
        self.p2.pack(side=LEFT)
        self.setpannle(self.img1, self.p1)
        self.setpannle(self.img2, self.p2)
        # Button(f3, text='主菜单', command=self.Main).pack(side=TOP)
        self.win.protocol('WM_DELETE_WINDOW', self.Main)
        # print('设置窗口内容')


class Window2(Window):
    '''
    风格迁移窗口
    '''

    def __init__(self, par):
        super().__init__()
        self.par = par
        self.img = np.zeros((400, 400, 3), np.uint8)
        self.img.fill(255)
        # 初始化dnn
        self.net = cv2.dnn.readNetFromTorch(model[4])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.p = None

    def Main(self):
        self.win.destroy()
        self.par.__init__()
        self.par.SetWindow()
        self.par.OpenWindow()

    def setpannle(self, img, p):
        current_image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=current_image)
        p.imgtk = imgtk
        p.config(image=imgtk)

    def model_select(self):
        path = askopenfilename(title='模型选择', filetypes=[('MODEL', '*.t7')])
        path.replace("/", "\\")
        self.net = cv2.dnn.readNetFromTorch(path)
        # self.setpannle(self.img2, self.p2)

    def image_select(self):
        path = askopenfilename(title='窗口标题', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        path.replace("/", "\\")
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (400, 400))
        show = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.setpannle(show, self.p)

    def trun(self):
        self.img = Style_Transfer(self.net, self.img)
        show = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.setpannle(show, self.p)

    def SetWindow(self):
        f1 = Frame(self.win)
        f1.pack(side=TOP)
        f2 = Frame(self.win)
        f2.pack(side=TOP)
        f3 = Frame(self.win)
        f3.pack(side=TOP)
        Button(f1, text='图片', command=self.image_select).pack(side=LEFT)
        Button(f1, text='模型', command=self.model_select).pack(side=LEFT)
        Button(f1, text='转化', command=self.trun).pack(side=LEFT)
        self.p = Label(f2)
        self.p.pack(side=LEFT)
        self.setpannle(self.img, self.p)
        # Button(f3, text='主菜单', command=self.Main).pack(side=TOP)
        self.win.protocol('WM_DELETE_WINDOW', self.Main)


class Window3(Window):
    '''
    手写数字识别窗口
    '''

    def __init__(self, par):
        super().__init__()
        # 初始化knn
        self.par = par
        self.knn = cv2.ml.KNearest_create()
        self.img = np.zeros((400, 400, 3), np.uint8)
        self.img.fill(255)
        self.p = None

    def file(self):
        path = askopenfilename(title='窗口标题', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        path.replace("/", "\\")
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (400, 400))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.setpannle(self.img, self.p)

    def rec(self):
        self.knn = Tr(self.knn)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        res = rec(self.knn, gray)
        print(res)

    def setpannle(self, img, p):
        current_image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=current_image)
        p.imgtk = imgtk
        p.config(image=imgtk)

    def Main(self):
        self.win.destroy()
        self.par.__init__()
        self.par.SetWindow()
        self.par.OpenWindow()

    def SetWindow(self):
        f1 = Frame(self.win)
        f1.pack(side=TOP)
        f2 = Frame(self.win)
        f2.pack(side=TOP)
        f3 = Frame(self.win)
        f3.pack(side=TOP)
        Button(f1, text='文件', command=self.file).pack(side=LEFT)
        Button(f1, text='识别', command=self.rec).pack(side=LEFT)
        self.p = Label(f2)
        self.p.pack(side=LEFT)
        self.setpannle(self.img, self.p)
        # Button(f3, text='主菜单', command=self.Main).pack(side=TOP)
        self.win.protocol('WM_DELETE_WINDOW', self.Main)


class Windowi(Window):
    def __init__(self, par):
        self.par = par
        super().__init__()
        self.img = np.zeros((400, 400, 3), np.uint8)
        self.img.fill(255)
        self.show = None
        self.p = None
        self.thr = 150

    def Main(self):
        self.win.destroy()
        self.par.__init__()
        self.par.SetWindow()
        self.par.OpenWindow()

    def setpannle(self, img, p):
        current_image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=current_image)
        p.imgtk = imgtk
        p.config(image=imgtk)


    def file(self):
        path = askopenfilename(title='窗口标题', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        path.replace("/", "\\")
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (400, 400))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.setpannle(self.img, self.p)

    def hd(self):
        show = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.setpannle(show, self.p)

    def hb(self):
        show = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        r, show = cv2.threshold(show, self.thr, 255, cv2.THRESH_BINARY_INV)
        self.setpannle(show, self.p)

    def hbs(self, thr):
        self.thr = int(thr)
        # print(type(self.thr))
        self.hb()


    def SetWindow(self):
        f1 = Frame(self.win)
        f2 = Frame(self.win)
        f3 = Frame(self.win)
        f1.pack(side=TOP)
        f2.pack(side=TOP)
        f3.pack(side=TOP)
        Button(f1, text='文件', command=self.file).pack(side=LEFT)
        Button(f1, text='灰度', command=self.hd).pack(side=LEFT)
        Button(f1, text='黑白', command=self.hb).pack(side=LEFT)
        self.p = Label(f2)
        self.p.pack(side=TOP)
        self.setpannle(self.img, self.p)
        s = Scale(f3, orient=HORIZONTAL, command=self.hbs).pack(side=TOP)
        self.win.protocol('WM_DELETE_WINDOW', self.Main)

        
class Windowa(Window):
    def __init__(self, par):
        self.par = par
        super().__init__()
        self.path = None
        self.show = None
        self.re = Recorder()

    def openPath(self):
        path_ = askopenfilename(title='选择一个.wav文件',
                                filetypes=[('Wave File', '*.wav')])  # 使用askdirectory()方法返回文件夹的路径
        if path_ == "":
            self.path.get()  # 当打开文件路径选择框后点击"取消" 输入框会清空路径，所以使用get()方法再获取一次路径
        else:
            path_ = path_.replace("/", "\\")  # 实际在代码中执行的路径为“\“ 所以替换一下
            self.path.set(path_)

    def rec(self):
        try:
            _path = self.path.get()
            text = self.re.rec_audio(_path)
            self.show.set(text)
        except Exception as e:
            print(e)

    def play(self):
        _path = self.path.get()
        self.re.play_audio(_path)

    def Main(self):
        self.win.destroy()
        self.par.__init__()
        self.par.SetWindow()
        self.par.OpenWindow()

    def SetWindow(self):
        self.win.title("这是一个窗口")
        self.win.geometry("600x100")  # 注意其中的x不是乘号！
        self.path = StringVar()
        self.path.set(os.path.abspath("."))
        self.show = StringVar()

        frame1 = Frame(self.win, relief=RAISED, height=200, width=400)
        frame1.pack(side=LEFT, fill=X)
        frame2 = Frame(self.win, relief=RAISED, height=200, width=200)
        frame2.pack(side=RIGHT, fill=X)
        ttk.Entry(frame1, textvariable=self.path, state="readonly").pack(side=LEFT, fill=Y)
        ttk.Entry(frame1, textvariable=self.show, state="readonly").pack(side=RIGHT)
        ttk.Button(frame1, text="文件选择", command=self.openPath).pack(side=LEFT, fill=Y)
        ttk.Button(frame1, text="文字识别", command=self.rec).pack(side=BOTTOM, fill=BOTH)
        ttk.Button(frame1, text="播放文件", command=self.play).pack(side=BOTTOM, fill=BOTH)

        ttk.Button(frame2, text="保存录音", command=self.re.save).pack(side=BOTTOM, fill=X)
        ttk.Button(frame2, text="结束录音", command=self.re.stop).pack(side=BOTTOM, fill=X)
        ttk.Button(frame2, text="开始录音", command=self.re.start).pack(side=BOTTOM, fill=X)
        self.win.protocol('WM_DELETE_WINDOW', self.Main)


class Window_C(Window):
    def __init__(self):
        super().__init__()
        self.win.title("主窗口")
        self.win.geometry("300x150")

    def NR(self):
        self.win.destroy()
        w = Window3(self)
        w.SetWindow()
        w.OpenWindow()
        print('手写数字识别')

    def ST(self):
        self.win.destroy()
        w = Window2(self)
        w.SetWindow()
        w.OpenWindow()
        print('风格迁移')

    def FM(self):
        self.win.destroy()
        w = Window1(self)
        w.SetWindow()
        w.OpenWindow()
        print('换脸')

    def YY(self):
        self.win.destroy()
        w = Windowa(self)
        w.SetWindow()
        w.OpenWindow()

    def ID(self):
        self.win.destroy()
        w = Windowi(self)
        w.SetWindow()
        w.OpenWindow()

    def SetWindow(self):
        tab = ttk.Notebook()
        tab.pack(side=LEFT)
        f1 = ttk.Frame(tab)
        f2 = ttk.Frame(tab)
        f3 = ttk.Frame(tab)
        f1.pack(side=LEFT)
        f2.pack(side=LEFT)
        f3.pack(side=LEFT)
        tab.add(f1, text='语音与文字')
        tab.add(f2, text='图像处理')
        tab.add(f3, text='简单应用')


        text1 = Text(f1, width=30, height=5, undo=True, autoseparators=False)
        text1.pack(side=TOP)
        text1.insert(INSERT, '语音与文字模块，包含了文字与语音的相互转换，并提供了录音和保存功能。')
        ttk.Button(f1, text='  语音', command=self.YY).pack(side=TOP)
        text2 = Text(f2, width=30, height=5, undo=True, autoseparators=False)
        text2.pack(side=TOP)
        text2.insert(INSERT, '图像处理功能，包括了彩色图向灰度图、二值图的转化功能。')
        ttk.Button(f2, text='  图像处理', command=self.ID).pack(side=TOP)
        text3 = Text(f3, width=30, height=5, undo=True, autoseparators=False)
        text3.pack(side=TOP)
        text3.insert(INSERT, '一些其他的简单的人工智能应用，提供了换脸、手写数字识别、风格迁移的功能。')
        ttk.Button(f3, text='手写数字识别', command=self.NR).pack(side=LEFT)
        ttk.Button(f3, text='风格迁移', command=self.ST).pack(side=LEFT)
        ttk.Button(f3, text='换脸', command=self.FM).pack(side=LEFT)

        text1.config(state=DISABLED)
        text2.config(state=DISABLED)
        text3.config(state=DISABLED)

        self.win.attributes('-alpha',0.95)
        # self.win.attributes('-transparentcolor', 'blue')
        # self.win['background'] = 'blue'

w1 = Window_C()
w1.SetWindow()
w1.OpenWindow()
