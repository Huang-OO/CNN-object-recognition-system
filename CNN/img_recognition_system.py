import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
# 创建画布需要的库
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# 导入绘图模块
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

window = tk.Tk()
window.title('img recognition system')
window.geometry('700x500')

ca1 = tk.Canvas(window, height=500, width=700)
img_file = tk.PhotoImage(file='bg.gif')
img = ca1.create_image(0, 0, anchor='nw', image=img_file)

num_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def recognition(file):
    global num_list
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('Model/saved_cnn.ckpt.meta')
    new_saver.restore(sess, 'Model/saved_cnn.ckpt')
    tf.get_default_graph().as_graph_def()
    a = sess.graph.get_tensor_by_name("input:0")
    y = sess.graph.get_tensor_by_name("output:0")
    b = sess.graph.get_tensor_by_name(("keep_prob:0"))

    image = Image.open(file)  # 打开特定一张图片
    image = image.resize((100, 100))  # 设置需要转换的图片大小
    image.save(file)
    img1 = plt.imread(file)
    img = img1 / 255
    img = np.array([img])

    result = sess.run(y, feed_dict={a: img, b: 0.5})
    i = 0
    for i in range(10):
        num_list[i] = result[0][i]
    print(num_list)
    fig = Figure(figsize=(2.95, 3), dpi=100)
    # 利用子图画图

    axc = fig.add_subplot(111)
    name_list = ['T恤', '兔子', '手表', '摩托车', '狗', '猫', '草莓', '西瓜', '轿车', '飞机']

    axc.barh(range(10), num_list, tick_label=name_list)
    for j in range(10):
        axc.text(+0.3, j - 0.3, num_list[j], ha='center', va='bottom', fontsize=11)

    # axc.bar(range(5),[0.1,0.2,0.3,0.1,0.3])

    # 创建画布控件

    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.

    canvas.draw()

    # 显示画布控件

    canvas.get_tk_widget().place(x=385, y=55)
    # print(result[0])

    if sess.run(tf.argmax(result, 1)) == 0:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一件T恤", "有{:.3f}%的概率是一件T恤".format(result[0][0] * 100))
        print("这有 {:.3f}%".format(result[0][0] * 100), "的概率是一件T恤")
        return "这有", result[0][0], "的概率是一件T恤"
    if sess.run(tf.argmax(result, 1)) == 1:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一只兔子", "有{:.3f}%的概率是一只兔子".format(result[0][1] * 100))
        print("这有 {:.3f}%".format(result[0][1] * 100), "的概率是一只兔子")
        return "这有", result[0][0], "的概率是一只兔子"
    if sess.run(tf.argmax(result, 1)) == 2:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一只手表", "有{:.3f}%的概率是一只手表".format(result[0][2] * 100))
        print("这有 {:.3f}%".format(result[0][2] * 100), "的概率是一只手表")
        return "这有", result[0][0], "的概率是一只手表"
    if sess.run(tf.argmax(result, 1)) == 3:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一辆摩托车", "有{:.3f}%的概率是一辆摩托车".format(result[0][3] * 100))
        print("这有 {:.3f}%".format(result[0][3] * 100), "的概率是一辆摩托车")
        return "这有", result[0][3], "的概率是一辆摩托车"
    if sess.run(tf.argmax(result, 1)) == 4:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一只狗", "有{:.3f}%的概率是一只狗".format(result[0][4] * 100))
        print("这有 {:.3f}%".format(result[0][4] * 100), "的概率是一只狗")
        return "这有", result[0][4], "的概率是一只狗"
    if sess.run(tf.argmax(result, 1)) == 5:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一只猫", "有{:.3f}%的概率是一只猫".format(result[0][5] * 100))
        print("这有 {:.3f}%".format(result[0][5] * 100), "的概率是一只猫")
        return "这有", result[0][5], "的概率是一只猫"
    if sess.run(tf.argmax(result, 1)) == 6:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是草莓", "有{:.3f}%的概率是草莓".format(result[0][6] * 100))
        print("这有 {:.3f}%".format(result[0][6] * 100), "的概率是草莓")
        return "这有", result[0][6], "的概率是草莓"
    if sess.run(tf.argmax(result, 1)) == 7:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是西瓜", "有{:.3f}%的概率是西瓜".format(result[0][7] * 100))
        print("这有 {:.3f}%".format(result[0][7] * 100), "的概率是西瓜")
        return "这有", result[0][7], "的概率是西瓜"
    if sess.run(tf.argmax(result, 1)) == 8:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是轿车", "有{:.3f}%的概率是轿车".format(result[0][8] * 100))
        print("这有 {:.3f}%".format(result[0][8] * 100), "的概率是轿车")
        return "这有", result[0][8], "的概率是轿车"

    else:
        re.delete(0, tk.END)
        re.insert(tk.END, "这个最可能是一只猫", "有{:.3f}%的概率是飞机".format(result[0][9] * 100))
        print("这有 {:.3f}%".format(result[0][9] * 100), "的概率是飞机")
        return "这有", result[0][9], "的概率是飞机"


files = None


def selectPath():
    global files
    path_ = askopenfilename(title='选择文件')
    path.set(path_)
    files = path_
    return files


def show_img(file):
    img = Image.open(file)
    img = img.resize((344, 394))
    photo = ImageTk.PhotoImage(img)
    img = tk.Label(window, image=photo)
    img.image = photo
    img.place(x=3, y=53)
    return file


path = tk.StringVar()
tk.Label(window, text='识别的图片', bg='yellow', width=13).place(x=19, y=10)
tk.Entry(window, textvariable=path).place(x=120, y=14)
tk.Button(window, text='选择文件', command=lambda: show_img(selectPath())).place(x=280, y=10)
# tk.Text(window,width = 49, height = 30).place(x = 0,y = 50)
ca = tk.Canvas(window, height=400, width=350, bg="white")
rect = ca.create_rectangle(2, 2, 351, 401)
ca.place(x=0, y=50)
tk.Button(window, text='识别图片', command=lambda: recognition(show_img(files)), width=49).place(x=0, y=460)
ca1.place(x=0, y=0)

tk.Label(window, text='识别结果', bg='yellow', width=13).place(x=500, y=10)

ca = tk.Canvas(window, height=400, width=300)
fig = Figure(figsize=(2.95, 3), dpi=100)
# 利用子图画图

axc = fig.add_subplot(111)
name_list = ['T恤', '兔子', '手表', '摩托车', '狗', '猫', '草莓', '西瓜', '轿车', '飞机']

axc.barh(range(10), num_list, tick_label=name_list)
# axc.bar(range(5),[0.1,0.2,0.3,0.1,0.3])

# 创建画布控件

canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.

canvas.draw()

# 显示画布控件

canvas.get_tk_widget().place(x=385, y=55)

re = tk.Listbox(window, font=('微软雅黑'), width=24, height=3)
re.place(x=386, y=358)

rect = ca.create_rectangle(2, 2, 300, 400)
ca.place(x=380, y=50)
button1 = tk.Button(window, text='退出', command=window.quit, width=42).place(x=380, y=460)

window.mainloop()
