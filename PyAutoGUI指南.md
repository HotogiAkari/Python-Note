<font size=5>PyAutoGUI指南</font>

>PyAutoGUI 是一个用于自动化计算机行为的Python库. 它可以用来操作鼠标和键盘, 比如移动鼠标, 点击按钮, 输入文本等. PyAutoGUI 还可以用来开发自动化工具, 比如自动回复聊天机器人, 自动游戏挂机等

<font color="red">PyAutoGUI无法对后台进行控制, 欲控制后台参考[UIAutoMation](UIAutoMation指南.md)</font>

目录
- [1. 预备知识](#1-预备知识)
  - [1. 故障保护功能](#1-故障保护功能)
- [2.](#2)
- [附录](#附录)


# 1. 预备知识

## 1. 故障保护功能

如果你的程序出现错误, 例如写错了一个坐标、流氓软件弹出了个窗口, 且无法使用键盘或鼠标关闭程序. 请在(默认为)0.1秒(根据你的pyautogui.PAUSE参数设置的数字, 你可以手动更改)内快速将鼠标移动到屏幕的四个角落之一. 如果你将`pyautogui.FAILSAFE`设置为`False`(默认为True), 以上防故障装置会关闭, 该方法失效, 现在唯一的方法是——注销. Windows操作系统按Ctrl+Alt+Delete屏幕就会变黑, 此时程序无法控制鼠标键盘, 单击中间的注销按钮即可注销. (注销操作会关闭计算机中的所有程序, 建议在注销前保存工作并关闭所有应用程序)

更改pyautogui.PAUSE的代码(0更改为想要的秒数, int类型或float类型)：


````py
import pyautogui as pg      # 通常把pyautogui导入为pg
pg.PAUSE = 0
````
这样, 当PyAutoGUI遇到错误时, 它会等待1秒后再尝试执行操作. 可以根据需要调整PAUSE的值

或者在PyAutoGUI程序执行过程中想要停止, 可以快速将鼠标移动到屏幕的四个角以中止程序, 默认存在的. 不想用可在PyAutoGUI代码执行之前插入

````py
pg.failsafe=false
````

# 2. 


# 附录

PyAutoGUI全部函数列表

1. 基本
   
| 函数名               | 功能                          |
| :------------------- | :---------------------------- |
| `pyautogui.size()`   | 返回包含分辨率的元组          |
| `pyautogui.PAUSE`    | 每个函数的停顿时间D, 默认0.1s |
| `pyautogui.FAILSAFE` | 是否开启防故障功能, 默认True  |

2. 键盘

| 函数名                                       | 功能               |
| :------------------------------------------- | :----------------- |
| `pyautogui.press('键盘字符')`                | 按下并松开指定按键 |
| `pyautogui.keyDown('键盘字符')`              | 按下指定按键       |
| `pyautogui.keyUp('键盘字符')`                | 松开指定按键       |
| `pyautogui.hotkey('键盘字符1', '键盘字符2')` | 按下多个指定键     |

3. 鼠标

| 函数名                                        | 功能                             |
| :-------------------------------------------- | :------------------------------- |
| `pyautogui.position()`                        | 返回当前鼠标当前位置的元组       |
| `pyautogui.moveTo(x,y,duration=1)`            | 按绝对位置移动鼠标并设置移动时间 |
| `pyautogui.moveRel(x_rel,y_rel,duration=4)`   | 按相对位置移动鼠标并设置移动时间 |
| `pyautogui.dragTo(x, y, duration=1)`          | 按绝对位置拖动鼠标并设置移动时间 |
| `pyautogui.dragRel(x_rel, y_rel, duration=4)` | 按相对位置拖动鼠标并设置移动时间 |
| `pyautogui.click(x, y)`                       | 鼠标点击指定位置，默认左键       |
| `pyautogui.click(x, y, button='left')`        | 鼠标单击左键                     |
| `pyautogui.click(x, y, button='right')`       | 鼠标单击右键                     |
| `pyautogui.click(x, y, button='middle')`      | 鼠标单击中间，即滚轮             |
| `pyautogui.doubleClick(10,10)`                | 鼠标左键双击指定位置             |
| `pyautogui.rightClick(10,10)`                 | 鼠标右键双击指定位置             |
| `pyautogui.middleClick(10,10)`                | 鼠标中键双击指定位置             |
| `pyautogui.scroll(10)`                        | 鼠标滚轮向上滚动10个单位         |

`press()`, `keyDowm()`, `keyUp()`, `hotKey()`支持的有效字符串列表如下

| 类别     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 字母     | `a`, `b`, `c`, `d`, `e`,`f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `p`, `q`, `r`, `s`, `t`, `u`, `v`, `w`, `x`, `y`, `z`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 数字     | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 符号     | `\t`, `\n`, `\r`, ` `, `!`, `"`, `#`, `$`, `%`, `&`, `'`, `(`, `)`, `*`, `+`, `,`, `-`, `.`, `/`, , `:`, `;`, `<`, `=`, `>`, `?`, `@`, `[`, `\\`, `]`, `^`, `_`, `` ` `` `{`, `\|`, `}`, `~`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| F键      | `f1`, `f2`, `f3`, `f4`, `f5`, `f6`, `f7`, `f8`, `f9`, `f10`, `f11`, `f12`, `f13`, `f14`, `f15`, `f16`, `f17`, `f18`, `f19`, `f20`, `f21`, `f22`, `f23`, `f24`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 数字键盘 | `num0`, `num1`, `num2`, `num3`, `num4`, `num5`, `num6`, `num7`, `num8`, `num9`,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 其它     | `accept`, `add`, `alt`, `altleft`, `altright`, `apps`, `backspace`, `browserback`, `browserfavorites`, `browserforward`, `browserhome`, `browserrefresh`, `browsersearch`, `browserstop`, `capslock`, `clear`, `convert`, `ctrl`, `ctrlleft`, `ctrlright`, `decimal`, `del`, `delete`, `divide`, `down`, `end`, `enter`, `esc`, `escape`, `execute`, `final`, `fn`, `hanguel`, `hangul`, `hanja`, `help`, `home`, `insert`, `junja`, `kana`, `kanji`, `launchapp1`, `launchapp2`, `launchmail`, `launchmediaselect`, `left`, `modechange`, `multiply`, `nexttrack`, `nonconvert`, , `numlock`, `pagedown`, `pageup`, `pause`, `pgdn`, `pgup`, `playpause`, `prevtrack`, `print`, `printscreen`, `prntscrn`, `prtsc`, `prtscr`, `return`, `right`, `scrolllock`, `select`, `separator`, `shift`, `shiftleft`, `shiftright`, `sleep`, `space`, `stop`, `subtract`, `tab`, `up`, `volumedown`, `volumemute`, `volumeup`, `win`, `winleft`, `winright`, `yen`, `command`, `option`, `optionleft`, `optionright` |

