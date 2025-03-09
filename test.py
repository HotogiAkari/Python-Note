import subprocess
import uiautomation as auto
import time

# 启动记事本
subprocess.Popen('notepad.exe')
time.sleep(2)  # 等待窗口加载

# 获取 Notepad 窗口
notepadWindow = auto.WindowControl(ClassName='Notepad')
notepadWindow.SetTopmost(True)
notepadWindow.SetFocus()

if not notepadWindow.Exists(5):
    print("❌ 记事本窗口未找到")
    exit()

notepadWindow.SetTopmost(True)
notepadWindow.SetFocus()

# **遍历 UI 结构**
def print_ui_structure(control, depth=0):
    print("  " * depth + f"[{control.ControlType}] Name: {control.Name}, Class: {control.ClassName}")
    for child in control.GetChildren():
        print_ui_structure(child, depth + 1)

print("🔍 遍历 UI 结构：")
print_ui_structure(notepadWindow)

print(notepadWindow.Name)
# 设置窗口前置
notepadWindow.SetTopmost(True)
# 获取 Notepad 窗口
edit = notepadWindow.PaneControl(ClassName="NotepadTextBox").TextControl(ClassName="RichEditD2DPT")
edit = notepadWindow.TextControl(ClassName="RichEditD2DPT")
if not edit.Exists(3):  
    print("🔍 方法1找不到，尝试方法2...")
    for pane in notepadWindow.PaneControl(foundIndex=1).GetChildren():
        if pane.ClassName == "NotepadTextBox":
            edit = pane.TextControl(ClassName="RichEditD2DPT")
            break
# 输入文本
if edit.Exists(3):
    print("✅ 找到编辑框！")
    edit.SendKeys('输入按键或内容')
    auto.SetClipboardText("TEST")
    edit.SendKeys('{Ctrl}v')
# # 通过标题栏查找名称为关闭的按钮
# notepadWindow.TitleBarControl(Depth=1).ButtonControl(searchDepth=1, Name='关闭').Click()
# 确认保存
auto.SendKeys('{ALT}s')
# 输入文件名，并快捷键点击保存
auto.SendKeys('自动保存{Ctrl}s')
# 如果弹出文件名冲突提示，则确认覆盖
auto.SendKeys('{ALT}y')
