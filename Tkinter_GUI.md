<font size=5>Tkinter GUI</font>

<font size=5>目录</font>

- [简介](#简介)
- [1. 创建主窗口](#1-创建主窗口)
- [2. 布局方式](#2-布局方式)
  - [1. `pack`方法](#1-pack方法)


# 简介

`tkinter` 模块可以创建图形用户界面, 是一种内建的标准模块, 不需要使用`pip`安装.

使用以下命令导入tkinter模块

````py
import tkinter
````

# 1. 创建主窗口

图形用户界面最外面是一个窗口对象, 称之为主窗口. 创建主窗口的方法如下

````py
win = tkinter.Tk()
````

主窗口包含以下几种方法

| 方法                    | 说明             | 示例                    |
| :---------------------- | :--------------- | :---------------------- |
| `geometry("WidexHigh")` | 设置主窗口的尺寸 | win.geometry("300x100") |
| `title(text)`           | 设置主窗口标题   | win.title("窗口标题")   |
| `mainloop()`            | 进入循环监听模式 | win.mainloop()          |

不一定要设置窗口大小, 如果没有设置标题,则默认为"tK"

主程序设置完成后, 应调用 `mainloop()` 方法, 让程序监听用户触发的事件, 一直到窗口关闭为止

示例

````py
import tkinter as tk
win = tk.Tk()
win.geometry("500x500")
win.title("Title")
win.mainloop()
````

运行结果如下

![图 0](images/%24%7Bpath%7D_%7B647B9E51-CFC1-48F0-A5CD-90018FB83462%7D.png)  

# 2. 布局方式

使用`Button`方法可以添加按钮, `Button`方法的使用方式如下

````py
btn = Button( master, parameter=value, ... )
````

- `master` -- 按钮的父容器
- `parameter` -- 按钮的参数
- `value` -- 参数的对应值

参数列表如下

| 参数               | 说明                                                                                                                                   |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `state`            | 按钮状态选项, 状态有DISABLED/NORMAL/ACTIVE                                                                                             |
| `activebackground` | 当鼠标放上去时, 按钮的背景色                                                                                                           |
| `activeforeground` | 当鼠标放上去时, 按钮的前景色                                                                                                           |
| `bd`               | 按钮边框的大小, 默认为 2 个像素                                                                                                        |
| `bg`               | 按钮的背景色                                                                                                                           |
| `fg`               | 按钮的前景色(按钮文本的颜色)                                                                                                           |
| `font`             | 文本字体, 文字字号, 文字字形. 字形有overstrike/italic/bold/underline                                                                   |
| `height`           | 按钮的高度, 如未设置此项, 其大小以适应按钮的内容(文本或图片的大小)                                                                     |
| `width`            | 按钮的宽度, 如未设置此项, 其大小以适应按钮的内容(文本或图片的大小)                                                                     |
| `image`            | 按钮上要显示的图片, 图片必须以变量的形式赋值给image, 图片必须是gif格式.                                                                |
| `justify`          | 显示多行文本的时候,设置不同行之间的对齐方式, 可选项包括LEFT, RIGHT, CENTER                                                             |
| `padx`             | 按钮在x轴方向上的内边距(padding), 是指按钮的内容与按钮边缘的距离                                                                       |
| `pady`             | 按钮在y轴方向上的内边距(padding)                                                                                                       |
| `relief`           | 边框样式, 设置控件显示效果, 可选的有: FLAT, SUNKEN, RAISED, GROOVE, RIDGE.                                                             |
| `wraplength`       | 限制按钮每行显示的字符的数量, 超出限制数量后则换行显示                                                                                 |
| `underline`        | 下划线. 默认按钮上的文本都不带下划线. 取值就是带下划线的字符串索引, 为 0 时, 第一个字符带下划线, 为 1 时, 第两个字符带下划线, 以此类推 |
| `text`             | 按钮的文本内容                                                                                                                         |
| `command`          | 按钮关联的函数, 当按钮被点击时, 执行该函数                                                                                             |

## 1. `pack`方法

`pack`方法默认从上往下摆放控件, 常见参数如下

| 参数     | 说明                                                                    |
| :------- | :---------------------------------------------------------------------- |
| `padx`   | 设置水平间距                                                            |
| `pady`   | 设置垂直间距                                                            |
| `side`   | 设置位置, 有`left` `right` `top` `bottom`四个参数                       |
| `expand` | 左右两端对齐, 参数为`0` `1`, `0`为不要分散, `1`为平均分配               |
| `fill`   | 是否填充, 参数有`x` `y` `both` `none`. `x`为填充的宽度, `y`为填充的高度 |

示例

````py
import tkinter as tk
win = tk.Tk()
win.geometry("500x500")
win.title("Title")

btn1 = tk.Button(win, width = 25, text = 'First Button')
btn1.pack()
btn2 = tk.Button(win, width = 25, text = 'Second Button')
btn2.pack()

win.mainloop()
````

运行结果如下

![图 1](images/%24%7Bpath%7D_pic_1740401745908.png)  
