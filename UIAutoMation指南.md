<font size=5>UIAutoMation指南</font>

<!--此处为文内使用的HTML, 请勿更改(以免造成内容错乱)-->

<style>
    .hid {
        color: black;
        background-color: black;
    }

    .hid:hover {
        color: white; /* 悬停时变为白色显示 */
    }
</style>

> `uiautomation`模块能直接兼容`pyautogui`支持这些操作，还能通过控件定位方式直接定位到目标控件的位置，而不需要自己去获取对应坐标位置。`uiautomation`模块不仅支持任意坐标位置截图，还支持目标控件的截图，缺点在于截取产生的图片对象难以直接与PIL库配合，只能导出文件后让PIL图像处理库重新读取。对于能够获取到其`ScrollItemPattern`对象的控件还可以通过`ScrollIntoView`方法进行视图定位，与游览器的元素定位效果几乎一致.常规的热键功能一般使用pynput实现，但`uiautomation模块`热键注册会比`pynput`更简单功能更强。`uiautomation`模块所支持的剪切板操作的功能也远远超过常规的专门用于剪切板复制粘贴的库。`uiautomation`模块能直接支持让python程序实现管理员提权

>uiautomation封装了微软UIAutomation API，支持自动化Win32，MFC，WPF，Modern UI(Metro UI), Qt, IE, Firefox( version<=56 or >=60), Chrome谷歌游览器和基于Electron开发的应用程序(加启动参数–force-renderer-accessibility也能支持UIAutomation被自动化).  
uiautomation只支持Python 3版本，依赖comtypes和typing这两个包，但Python不要使用3.7.6和3.8.1这两个版本，comtypes在这两个版本中不能正常工作

<font size=5>目录</font>

- [1. 基本原理](#1-基本原理)


# 1. 基本原理

**UIAutomation的工作原理：**

UIAutomation操作程序时会给程序发送WM_GETOBJECT消息，如果程序处理WM_GETOBJECT消息，实现UI Automation Provider,并调用函数

UiaReturnRawElementProvider(HWND hwnd,WPARAM wparam,LPARAM lparam,IRawElementProviderSimple *el)，此程序就支持UIAutomation。

IRawElementProviderSimple 就是 UI Automation Provider，包含了控件的各种信息，如Name，ClassName，ContorlType，坐标等。

UIAutomation 根据程序返回的 IRawElementProviderSimple，就能遍历程序的控件，得到控件各种属性，进行自动化操作。若程序没有处理WM_GETOBJECT或没有实现UIAutomation Provider，UIAutomation则无法识别这些程序内的控件，不支持自动化。

<style=hid>很多DirectUI程序没有实现UIAutomation Provider，所以不支持自动化