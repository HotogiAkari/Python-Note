
import subprocess
import uiautomation as auto
 
subprocess.Popen('notepad.exe')# 从桌面的第一层子控件中找到记事本程序的窗口WindowControl
notepadWindow = auto.WindowControl(searchDepth=1, ClassName='Notepad')
print(notepadWindow.Name)# 设置窗口前置
notepadWindow.SetTopmost(True)