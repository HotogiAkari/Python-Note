import subprocess
import uiautomation as auto
 
subprocess.Popen('notepad.exe')# 首先从桌面的第一层子控件中找到记事本程序的窗口WindowControl，再从这个窗口查找子控件
notepadWindow = auto.WindowControl(searchDepth=1, ClassName='Notepad')
print(notepadWindow.Name)# 设置窗口前置
notepadWindow.SetTopmost(True)
# 获取 Notepad 窗口
notepad = auto.WindowControl(searchDepth=1, ClassName='Notepad')
notepad.SetFocus()
# 输入文本
edit = notepad.EditControl()
edit.GetValuePattern().SetValue("输入内容")
edit.SendKeys('输入按键或内容')
auto.SetClipboardText("TEST")
edit.SendKeys('{Ctrl}v')
# 获取文本
print("编辑框内容：",edit.GetValuePattern().Value)
# # 通过标题栏查找名称为关闭的按钮
notepadWindow.TitleBarControl(Depth=1).ButtonControl(searchDepth=1, Name='关闭').Click()
# 确认保存
auto.SendKeys('{ALT}s')
# 输入文件名，并快捷键点击保存
auto.SendKeys('自动保存{Ctrl}s')
# 如果弹出文件名冲突提示，则确认覆盖
auto.SendKeys('{ALT}y')