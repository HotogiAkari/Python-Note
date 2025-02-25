# Python基础笔记

<!--此处为文内使用的HTML, 请勿更改(以免造成内容错乱)-->

<style>
    .hidden-text {
        color: black;
        background-color: black;
    }

    .hidden-text:hover {
        color: white; /* 悬停时变为白色显示 */
    }
</style>

## 1. 目录
- [Python基础笔记](#python基础笔记)
  - [1. 目录](#1-目录)
  - [2. 代码](#2-代码)
    - [1. 循环](#1-循环)
      - [1. `while`$~~$循环](#1-while循环)
      - [2. `for`$~~$循环](#2-for循环)
      - [3. `continue` , `pass` \& `break`](#3-continue--pass--break)
    - [2. 条件](#2-条件)
      - [1.`if`$~~$和$~~$`else`$~~$条件语句](#1if和else条件语句)
    - [3. 迭代](#3-迭代)
      - [1. `range`$~~$迭代器](#1-range迭代器)
    - [4. 函数\&方法](#4-函数方法)
      - [1. `def`$~~$定义](#1-def定义)
      - [2. 序列函数](#2-序列函数)
        - [1. `cmp()` 方法](#1-cmp-方法)
        - [2. `len()` 方法](#2-len-方法)
        - [3. `max()` 方法\&`min()` 方法](#3-max-方法min-方法)
        - [4. 序列数据类型转换](#4-序列数据类型转换)
    - [5. 文件读写](#5-文件读写)
      - [1. $~$`open` $~~$打开文件](#1-open-打开文件)
    - [6. 模块](#6-模块)
      - [1. 创建模块](#1-创建模块)
      - [2. `import`模块载入](#2-import模块载入)
      - [3. 常见模块索引](#3-常见模块索引)
    - [7. 其它](#7-其它)
      - [1. 异常处理](#1-异常处理)
      - [2. `zip()`](#2-zip)
      - [3. `lambda` 匿名函数](#3-lambda-匿名函数)
      - [4. `map()`](#4-map)
      - [5. `copy` \& `deepcopy`](#5-copy--deepcopy)
  - [3. 容器类型](#3-容器类型)
    - [1. 序列](#1-序列)
      - [1. 列表](#1-列表)
      - [2. 元组](#2-元组)
    - [2. 字典](#2-字典)
  - [4. 面向对象编程](#4-面向对象编程)
    - [1. 对象和类](#1-对象和类)
    - [数据封装](#数据封装)

## 2. 代码

### 1. 循环

#### 1. `while`$~~$循环

`while` 用于条件循环.  满足条件时会一直循环

**例子**

````py
condition = 1
while comdition < 3:       #注意此处为冒号: 不是分号;
    print(condition)        #注意缩进
    condition ++
````

输出如下

```
1
2
3
```

#### 2. `for`$~~$循环

`for` 循环可以遍历任何序列的项目, 如一个列表或者一个字符串

**例子**

````py
fruits = ['banana', 'apple',  'mango']
for fruit in fruits:        # 注意冒号和缩进
   print ('当前水果: %s'% fruit)
 
print ("Good bye!")
````

输出如下

```
当前水果: banana
当前水果: apple
当前水果: mango
Good bye!
```

#### 3. `continue` , `pass` & `break`

`continue`用于略过循环内剩余内容,回到循环开头
`pass`是空语句, 不做任何事情, 一般用做占位语句
`break`用于跳出循环,不执行后面的语句

````py
while True:
	a = input('type a number')
	if a == '1':
		break
	elif a == '0':
		pass
	else:
		continue
	print('still in while')

print('finish run')
````

输出如下

```
type a number
>>> 2
type a number
>>> 0
still in while
type a number
>>> 1
finish run
```

### 2. 条件

#### 1.`if`$~~$和$~~$`else`$~~$条件语句

**例子**

````py
x = 1
y = 2
z = 0
if x < y > z:
    print('x is less than y, and y is large than z')
elif x < y < z:
    print('x is less than y, and y is less than z')
else:
    print('x isn\'t less than y, or y isn\'t less than z')
````

输出如下

```
x is less than y, and y is less than 
```

### 3. 迭代

#### 1. `range`$~~$迭代器

`range(a,b,c)` 相当于一个包含整数 `a` , `a+1` …… `b-1` 的`list`

**例子**

````py
for i in range (1,10,2):
    print (i)
````

输出如下

```
1
3
5
7
9
```

**注意**  `range` 中第三个数[^1]可以不给出,默认为1

### 4. 函数&方法

#### 1. `def`$~~$定义

**例子**

````py
# 定义函数
def function(a, b):
    a++
    b++
    c = a + b
    print('c is ' c)

# 执行函数
function(1, 2)
````

输出如下

```
c is 5
```

#### 2. 序列函数

序列类函数基本相同,故统一介绍.  
关于序列,参考[序列类型](#3-序列类型)

##### 1. `cmp()` 方法

`cmp()`方法比较两个序列并返回值

**语法**

````py
cmp(array1, array2)
````

**参数**

- `array1`		比较的列表
- `array2`		比较的列表

**返回值**

1. 比较时, 若 `array1` > `array2` 输出 1, `array1` < `array2` 则输出 -1

2. 如果比较的元素是同类型的, 则比较其值, 返回结果

3. 如果两个元素不是同一种类型, 则检查它们是否是数字
	- 如果是数字, 执行必要的数字强制类型转换, 然后比较
	- 如果有一方的元素是数字, 则另一方的元素"大"(数字是"最小的")
	- 否则, 通过类型名字的字母顺序进行比较

4. 如果有一个列表首先到达末尾, 则另一个长一点的列表"大"

5. 如果我们用尽了两个列表的元素而且所有元素都是相等的, 则返回一个 0

**例子**

````py
list1, list2 = [123, 'xyz'], [456, 'abc']

print cmp(list1, list2);
print cmp(list2, list1);
list3 = list2 + [786];
print cmp(list2, list3)
````

输出如下

```
-1
1
-1
```

##### 2. `len()` 方法

len()方法返回序列元素个数

**语法**

````py
len(array)
````

**参数**

- `array`		要计算元素的序列

**返回值**

返回列表元素个数

**例子**

````py
list1, list2 = [123, 'xyz', 'zara'], [456, 'abc']

print "First list length : ", len(list1);
print "Second list length : ", len(list2);
````

输出如下

```
First list length :  3
Second lsit length :  2
```

##### 3. `max()` 方法&`min()` 方法

max()方法返回序列最大值

**语法**

````py
max(array)
min(array)
````

**参数**

- `array`		返回列表元素中的最大值(最小值)

**返回值**

返回列表元素中的最大值

**例子**

````py
list1, list2 = ['123', 'xyz', 'zara', 'abc'], [456, 700, 200]

print "Max value element : ", max(list1);
print "Min value element : ", min(list2);
````

输出如下

```
Max value element :  zara
Min value element :  200
```

##### 4. 序列数据类型转换

**语法**

````py
list(tup)
tuple(iterable)
````

**参数**

- tup			要转换为列表的元组
- iterable		要转换为元组的可迭代序列

**返回值**

`list`:			返回列表
`tuple`:		返回元组

**例子**

````py
aTuple = (123, 'abc');
aList = list(aTuple)
 
print "List elements : ", aList

aList = [123, 'xyz', 'abc'];
aTuple = tuple(aList)
 
print "Tuple elements : ", aTuple
````

输出如下

```
List elements : [123, 'abc']
Tuple elements :  (123, 'xyz', 'abc')
```

### 5. 文件读写

#### 1. $~$`open` $~~$打开文件

`open()`函数用于打开一个文件, 创建一个`file`对象, 相关的方法才可以调用它进行读写.

**格式**

````py
open('NAME'[,'Mode'[,BUFFERING]])
````

- `NAME` : 一个包含了你要访问的文件名称的字符串值. 

- `Mode` : `Mode` 决定了打开文件的模式: 只读, 写入, 追加等. 所有可取值见如下的完全列表. 这个参数是非强制的, 默认文件访问模式为只读(r). 

- `BUFFERING` : 如果 `BUFFERING` 的值被设为 0, 就不会有寄存. 如果 `BUFFERING` 的值取 1, 访问文件时会寄存行. 如果将 `BUFFERING` 的值设为大于 1 的整数, 表明了这就是的寄存区的缓冲大小. 如果取负值, 寄存区的缓冲大小则为系统默认
  

`Mode`内有以下几种模式:

| 模式  | 功能                                                                                                                                                             |
| :---- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `t`   | 文本模式 (默认)                                                                                                                                                  |
| `x`   | 写模式, 新建一个文件, 如果该文件已存在则会报错                                                                                                                   |
| `b`   | 二进制模式                                                                                                                                                       |
| `+`   | 打开一个文件进行更新(可读可写)                                                                                                                                   |
| `U`   | 通用换行模式 (不推荐)                                                                                                                                            |
| `r`   | 以只读方式打开文件. 文件的指针将会放在文件的开头. 这是默认模式.                                                                                                  |
| `rb`  | 以二进制格式打开一个文件用于只读. 文件指针将会放在文件的开头. 这是默认模式. 一般用于非文本文件如图片等                                                           |
| `r+`  | 打开一个文件用于读写. 文件指针将会放在文件的开头                                                                                                                 |
| `rb+` | 以二进制格式打开一个文件用于读写. 文件指针将会放在文件的开头. 一般用于非文本文件如图片等                                                                         |
| `w`   | 打开一个文件只用于写入. 如果该文件已存在则打开文件, 并从开头开始编辑, 即原有内容会被删除. 如果该文件不存在, 创建新文件                                           |
| `wb`  | 以二进制格式打开一个文件只用于写入. 如果该文件已存在则打开文件, 并从开头开始编辑, 即原有内容会被删除. 如果该文件不存在, 创建新文件. 一般用于非文本文件如图片等   |
| `w+`  | 打开一个文件用于读写. 如果该文件已存在则打开文件, 并从开头开始编辑, 即原有内容会被删除. 如果该文件不存在, 创建新文件                                             |
| `wb+` | 以二进制格式打开一个文件用于读写. 如果该文件已存在则打开文件, 并从开头开始编辑, 即原有内容会被删除. 如果该文件不存在, 创建新文件. 一般用于非文本文件如图片等     |
| `a`   | 打开一个文件用于追加. 如果该文件已存在, 文件指针将会放在文件的结尾. 也就是说, 新的内容将会被写入到已有内容之后. 如果该文件不存在, 创建新文件进行写入             |
| `ab`  | 以二进制格式打开一个文件用于追加. 如果该文件已存在, 文件指针将会放在文件的结尾. 也就是说, 新的内容将会被写入到已有内容之后. 如果该文件不存在, 创建新文件进行写入 |
| `a+`  | 打开一个文件用于读写. 如果该文件已存在, 文件指针将会放在文件的结尾. 文件打开时会是追加模式. 如果该文件不存在, 创建新文件用于读写                                 |
| `ab+` | 以二进制格式打开一个文件用于追加. 如果该文件已存在, 文件指针将会放在文件的结尾. 如果该文件不存在, 创建新文件用于读写                                             |

---

file 对象方法
- `file.read([size])`: `size` 从文件读取指定的字节数, 如果未给定或为负则读取所有. 默认为 -1, 表示读取整个文件.f.read()读到文件尾时返回""(空字串).
  
- `file.readline()`: 返回一行.

- `file.readlines([size])`: 返回包含size行的列表, size 未指定则返回全部行.

- `for line in f:
    print(line)`: 逐行读取文件, 并将每一行内容输出到屏幕上.

- `f.write("hello\n")`: 在文末写入数据. 如果要写入字符串以外的数据,先将其转换为字符串.

- `f.tell()`: 返回一个整数,表示当前文件指针的位置(就是到文件头的字节数).

- `f.seek(偏移量,[起始位置])`: 用来移动文件指针.

  - 偏移量: 单位为字节, 可正可负
  - 起始位置: 0 - 文件头, 默认值; 1 - 当前位置; 2 - 文件尾
- `f.close()` 关闭文件

**例子**

假设文件`test.txt`内容为

> wasd

````py
a = open('test.txt','t')
a.read()
````

输出如下

```
wasd
```

### 6. 模块

#### 1. 创建模块

模块为一个单独的`.py`文件,如下是一个自定义的模块 `support.py`

````py
def print_name( par ):
   print "Hello : ", par
   return
````

调用该模块里函数的方法如下

````py
import support						# 导入模块

support.print_name("Koushaku")		# 使用模块内函数
````

输出如下

```
Hello : Koushaku
```

#### 2. `import`模块载入

import语句用来导入其他 python 文件.  
一个模块只会被导入一次, 不管你执行了多少次import

````py
import module1[, module2[,... moduleN]]
````

例如如要引用模块 `time`, 就可以在文件最开始的地方用 `import time` 来引入. 在调用 `time` 模块中的函数时, 必须这样引用

```
time.localtime
```

<font size="5">`from…import` 语句</font>

from 语句可以从模块中导入一个指定的部分到当前命名空间中

````py
from modname import name1[, name2[, ... nameN]]
````

例如导入模块 `fib` 的 `fibonacci` 函数

````py
from fib import fibonacci
````

这个声明不会把整个 fib 模块导入到当前的命名空间中, 它只会将 fib 里的 fibonacci 单个引入到执行这个声明的模块的全局符号表

<font size="5">拓展(`import`的用法)</font>

<font size="4">1. `import module_name`</font>

import 后直接接模块名时, Python 会在以下两个地方寻找这个模块

- sys.path(通过运行代码`import sys` `print(sys.path)`查看).一般安装的 Python 库的目录都可以在 sys.path 中找到(前提是要将 Python 的安装目录添加到电脑的环境变量), 所以对于安装好的库, 我们直接 import 即可
- 运行文件所在的目录

❗ 此方法导入原有的 `sys.path` 中的库没有问题. 但是, 最好不要用上述方法导入同目录下的文件!<span class="hidden-text">若用该方法导入同文件夹其他文件, 从不同文件夹导入该模块的文件无法导入该模组内导入的其它文件</span>

<font size="4">2. `from package_name import module_name`</font>

在 Pythonproject目录下新建一个目录 package, 在 package 中新建文件 support.py并写入

````py
def check_support ():
	print('support is ready')
````

若要在main.py中导入support.py,需要执行以下操作

````py
from package import support
support.check_support
````

一般把模块组成的集合称为包(package). 与第一种写法类似, Python 会在 sys.path 和运行文件目录这两个地方寻找包, 然后导入包中名为 module_name 的模块

以上两种写法属于绝对导入, 如果是_非运行入口_文件则需要相对导入.

<font size="4">3. 相对导入</font>

相对导入仍使用`from package_name import module_name`

- from . import module_name. 导入和自己同目录下的模块
- from .package_name import module_name. 导入和自己同目录的包的模块
- from .. import module_name. 导入上级目录的模块
- from ..package_name import module_name. 导入位于上级目录下的包的模块  

每多一个`.`就多往上一层目录

<font size="4">4. `import`其他简单但实用的用法</font>

- `import moudle_name as alias`$~~~~~~~~$有些 module_name 比较长, 之后写它时较为麻烦, 或者 module_name 会出现名字冲突, 可以用 as 来给它改名, 如`import numpy as np`. 
- `from module_name import function_name, variable_name, class_name`$~~~~~~~~$上面导入的都是整个模块, 有时候我们只想使用模块中的某些函数、某些变量、某些类, 用这种写法就可以了. 使用逗号可以导入模块中的多个元素. 
- 有时候导入的元素很多, 可以使用反斜杠来换行, 官方推荐使用括号. 

#### 3. 常见模块索引

1. [`numpy` & `pandas`<span class=hidden-text>还没做好</span>](Numpy&Pandas.md)
2. [`Threading`多线程](Threading多线程.md)
3. [`Tkinter` GUI](Tkinter_GUI.md)

### 7. 其它

#### 1. 异常处理

程序在运行的时候, 如果python解释器遇到一个错误, 会停止程序的执行, 并且提示一些错误的信息, 这就是异常  
我们在程序开发的时候, 很难将所有的特殊情况都处理, 通过异常捕获可以针对突发事件做集中处理, 从而保证程序的健壮性和稳定性

在程序开发中, 如果对某些代码的执行不能确定(程序语法完全正确)可以增加`try`来捕获异常

<font size="5">**`try`捕获异常**</font>  

`try/except`语句用来检测`try`语句块中的错误, 从而让`except`语句捕获异常信息并处理, 这样发生异常时程序不会结束
`try`:尝试执行的代码  
`except`:出现错误的处理

````py
try:
	# 不能确定正确执行的代码
	num = int(input('请输入一个数字:'))
except :
	print('请输入正确的数字')

print(num)
try:
	<语句>		#运行别的代码
except <名字>
	<语句>        #如果在try部份引发了'name'异常
except <名字>,<数据>:
	<语句>        #如果引发了'name'异常, 获得附加的数据
else:
	<语句>        #如果没有异常发生
````

<font size="5">使用`except`而不带任何异常类型</font>

**示例**

````py
try:
	fh = open("testfile", "w")
	fh.write("这是一个测试文件, 用于测试异常!!")
except IOError:
	print "Error: 没有找到文件或读取文件失败"
else:
	print "内容写入文件成功"
	fh.close()
````

出现异常时输出如下

```
$ python test.py 
Error: 没有找到文件或读取文件失败
```

无异常时输出如下

```
$ python test.py 
内容写入文件成功
$ cat testfile       # 查看写入的内容
这是一个测试文件, 用于测试异常!!
```

**以上方式try-except语句捕获所有发生的异常. 但这不是一个很好的方式, 我们不能通过该程序识别出具体的异常信息. 因为它捕获所有的异常**

<font size="5">使用`except`而带多种异常类型</font>

可以使用相同的except语句来处理多个异常信息

````py
try:
    正常的操作
except(Exception1[, Exception2[,...ExceptionN]]):
   发生以上多个异常中的一个, 执行这块代码
else:
    如果没有异常执行这块代码
````

<font size="5">`try-finally` 语句</font>

try-finally 语句无论是否发生异常都将执行最后的代码

````py
try:
<语句>
finally:
<语句>    #退出try时总会执行
raise
````

**示例**
````py
try:
    fh = open("testfile", "w")
    fh.write("这是一个测试文件, 用于测试异常!!")
finally:
    print "Error: 没有找到文件或读取文件失败"
````

出现异常时输出如下

```
$ python test.py 
Error: 没有找到文件或读取文件失败
```

也可以写成如下方式

````py
try:
    fh = open("testfile", "w")
    try:
        fh.write("这是一个测试文件, 用于测试异常!!")
    finally:
        print "关闭文件"
        fh.close()
except IOError:
    print "Error: 没有找到文件或读取文件失败"
````

<font size="5">异常的参数</font>

一个异常可以带上参数, 可作为输出的异常信息参数  
可以通过except语句来捕获异常的参数

````py
try:
    正常的操作
except ExceptionType, Argument:
    可以在这输出 Argument 的值...
````

变量接收的异常值通常包含在异常的语句中. 在元组的表单中变量可以接收一个或者多个值.   
元组通常包含错误字符串, 错误数字, 错误位置. 

示例

````py
# 定义函数
def temp_convert(var):
    try:
        return int(var)
    except ValueError, Argument:
        print "参数没有包含数字\n", Argument

# 调用函数
temp_convert("xyz")
````

输出如下

```
$ python test.py 
参数没有包含数字
invalid literal for int() with base 10: 'xyz'
```

<font size="5">触发异常</font>

可以使用`raise`语句主动触发异常

````py
raise [Exception [, args [, traceback]]]
````

- `Exception` 异常的类型(例如NameError)参数标准异常中任一种  
- `args` 主动提供的异常参数
- `traceback`是可选参数(很少使用), 如果存在, 是跟踪异常对象

一个异常可以是一个字符串, 类或对象.  Python的内核提供的异常, 大多数都是实例化的类, 这是一个类的实例的参数

示例

````py
def functionName( level ):
    if level < 1:
        raise Exception("Invalid level!", level)
        # 触发异常后, 后面的代码就不会再执行
````

**注意**: 为了能够捕获异常, `except`语句必须有用相同的异常来抛出类对象或者字符串.   
例如捕获以上异常, "except"语句如下所示

````py
try:
    正常逻辑
except Exception,err:
    触发自定义异常    
else:
    其余代码
````

示例

````py
# 定义函数
def mye( level ):
    if level < 1:
        raise Exception,"Invalid level!"
        # 触发异常后, 后面的代码就不会再执行
try:
    mye(0)            # 触发异常
except Exception,err:
    print 1,err
else:
    print 2
````

输出如下

```
$ python test.py 
1 Invalid level!
```

<font size="5">自定义异常</font>

通过创建一个新的异常类, 程序可以命名它们自己的异常. 异常应该是典型的继承自`Exception`类, 通过直接或间接的方式

以下为与`RuntimeError`相关的实例, 实例中创建了一个类, 基类为`RuntimeError`, 用于在异常触发时输出更多的信息.   
在`try`语句块中, 用户自定义的异常后执行`except`块语句, 变量 e 是用于创建`Networkerror`类的实例

````py
class Networkerror(RuntimeError):
    def __init__(self, arg):
        self.args = arg
````

你定义以上类后, 可以触发该异常

````py
try:
    raise Networkerror("Bad hostname")
except Networkerror,e:
    print e.args
````

#### 2. `zip()`

`zip()` 函数用于将可迭代的对象作为参数, 将对象中对应的元素打包成一个个元组, 然后返回由这些元组组成的列表

如果各个迭代器的元素个数不一致, 则返回列表长度与最短的对象相同, 利用`*` 号操作符, 可以将元组解压为列表

>*zip 方法在 Python 2 和 Python 3 中的不同: 在 Python 3.x 中为了减少内存, zip() 返回的是一个对象. 如需展示列表, 需手动 list() 转换*

**语法**

````py
zip([iterable, ...])
````
**参数**

- `iterable` -- 可迭代对象(如列表、元组、字符串等)

**返回值**

返回元组列表

示例

````py
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 返回一个对象
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() 转换为列表
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]

>>> a1, a2 = zip(*zip(a,b))          # 与 zip 相反, zip(*) 可理解为解压, 返回二维矩阵式
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
>>>
````

#### 3. `lambda` 匿名函数

Python 使用 lambda 来创建匿名函数.   
lambda 函数是一种小型, 匿名的内联函数, 它可以具有任意数量的参数, 但只能有一个表达式.   
匿名函数不需要使用 def 关键字定义完整函数.   
lambda 函数通常用于编写简单, 单行的函数, 通常在需要函数作为参数传递的情况下使用, 例如在 `map()` `filter()` `reduce()` 等函数中. 

**特点**

- `lambda` 函数是匿名的, 它们没有函数名称, 只能通过赋值给变量或作为参数传递给其他函数来使用
- `lambda` 函数通常只包含一行代码, 这使得它们适用于编写简单的函数

**语法**

````py
lambda arguments: expression
````

**参数**

- arguments -- 参数列表, 可以包含零个或多个参数, 但必须在冒号`:`前指定
- expression -- 一个表达式, 用于计算并返回函数的结果
  
示例

没有参数的lambda函数

````py
f = lambda: "Hello, world!"
print(f())  # 输出: Hello, world!
````

输出如下

```
Hello, world!
```

使用 lambda 创建匿名函数, 设置一个函数参数 a, 函数计算参数 a 加 10, 并返回结果

````py
x = lambda a : a + 10
print(x(5))
````

输出如下

```
15
```

`lambda` 函数也可以设置多个参数, 参数使用逗号 `,` 隔开  
以下实例使用 `lambda` 创建匿名函数, 函数参数 `a` 与 `b` 相乘, 并返回结果

````py
x = lambda a, b : a * b
print(x(5, 6))
````

输出如下

```
30
```

以下实例使用 lambda 创建匿名函数, 函数参数 a、b 与 c 相加, 并返回结果

````py
x = lambda a, b, c : a + b + c
print(x(5, 6, 2))
````

输出如下

```
13
```

`lambda` 函数通常与内置函数如 [`map()`]() [`filter()`]() 和 [`reduce()`]() 一起使用, 以便在集合上执行操作

````py
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # 输出: [1, 4, 9, 16, 25]
````

输出如下

```
[1, 4, 9, 16, 25]
```

#### 4. `map()`

`map()` 会根据提供的函数对指定序列做映射. 

第一个参数 `function` 以参数序列中的每一个元素调用 `function` 函数, 返回包含每次 `function` 函数返回值的新列表

**语法**

````py
map(function, iterable, ...)
````

**参数**

- function -- 函数
- iterable -- 一个或多个序列

**返回值**

返回迭代器

示例

````py
>>> def square(x) :         # 计算平方数
	    return x ** 2 
>>> map(square, [1,2,3,4,5])    # 计算列表各个元素的平方
<map object at 0x100d3d550>     # 返回迭代器
>>> list(map(square, [1,2,3,4,5]))   # 使用 list() 转换为列表
[1, 4, 9, 16, 25]
>>> list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]
>>>
````

#### 5. `copy` & `deepcopy`

使用`deepcopy` 需要引入`copy`模块

- `copy` -- 拷贝父对象, 不会拷贝对象的内部的子对象
- `deepcopy` -- `copy` 模块的 `deepcopy` 方法, 完全拷贝了父对象及其子对象

直接赋值

````py
>>>a = {1: [1,2,3]}
>>> b = a
>>> a, b
({1: [1, 2, 3]}, {1: [1, 2, 3]})
>>> a[1].append(4)
>>> a, b
({1: [1, 2, 3, 4]}, {1: [1, 2, 3, 4]})
````

直接赋值使两个变量索引同一个位置,因此改变其中一个变量,另一个也会改变

浅拷贝(需要引入copy模块)

````py
>>> a = [1, 2, [3, 4]]
>>> d = copy.copy(a)
>>> id(a) == id(d)
False
>>> id(a[2]) == id(d[2])
True
````
浅拷贝只拷贝一层<spaan class="hidden-text">列表中的列表被直接赋值</span>

深拷贝<span class="hidden-text">需要引入copy模块</span>

````py
>>>import copy
>>> c = copy.deepcopy(a)
>>> a, c
({1: [1, 2, 3, 4]}, {1: [1, 2, 3, 4]})
>>> a[1].append(5)
>>> a, c
({1: [1, 2, 3, 4, 5]}, {1: [1, 2, 3, 4]})
````

深拷贝完全复制内容,修改一个变量并不会影响另一个变量



## 3. 容器类型

### 1. 序列

  序列是Python中最基本的数据结构. 序列中的每个元素都分配一个数字, 即位置或索引. 第一个索引是0,  第二个索引是1,  依此类推.  
  Python有6个序列的内置类型, 但最常见的是列表和元组.  
  序列都可以进行的操作包括索引, 切片, 加, 乘, 检查成员.  
此外, Python已经内置确定序列的长度以及确定最大和最小的元素的方法.  

#### 1. 列表

<b>列表(list)</b>是最常用的Python数据类型,它可以作为一个方括号内的逗号分隔值出现.  
列表的数据项不需要具有相同的类型.

创建一个列表, 只要把逗号分隔的不同的数据项使用方括号括起来即可
````py
list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]
````

使用将列表作为数据项  可以创建多维列表

````py
list4 = [[1, 2, 3],
		 [4, 5, 6]]
````

与字符串的索引一样, 列表索引从0开始. 列表可以进行截取/组合等

<font size="5">访问列表</font>

使用下标索引来访问列表中的值, 也可以使用方括号的形式截取字符

````py
list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5, 6, 7 ]
 
print "list1[0]: ", list1[0]
print "list2[1:5]: ", list2[1:5]
print(list2[1])
print(list2[1][2])
````

输出如下

```
list1[0]:  physics
list2[1:5]:  [2, 3, 4, 5]
2
6
```

<font size="5">更新列表</font>

可以对列表的数据项进行修改或更新, 也可以使用append()方法来添加列表项

````py
list = []				# 空列表
list.append('Google')	# 使用 append() 添加元素
list.append('Runoob')
print list				# ['Google', 'Runoob']
````

<font size="5">删除列表元素</font>

可以使用 del 语句来删除列表的元素

````py
list1 = ['physics', 'chemistry', 1997, 2000]
 
print list1
del list1[2]
print "After deleting value at index 2 : "
print list1
````

输出如下

```
['physics', 'chemistry', 1997, 2000]
After deleting value at index 2 :
['physics', 'chemistry', 2000]
```

<font size="5">列表脚本操作符</font>

列表对 + 和 * 的操作符与字符串相似. + 号用于组合列表, * 号用于重复列表

| 代码         | 描述                 | 表达式                    | 结果           |
| :----------- | :------------------- | :------------------------ | :------------- |
| `len`        | 长度                 | len([1, 2, 3])            | 3              |
| `+`          | 组合                 | [1, 2] + [4, 5]           | [1, 2, 3, 4]   |
| `*`          | 重复                 | ['Hi!'] * 2               | ['Hi!', 'Hi!'] |
| `in`         | 元素是否存在于列表中 | 3 in [1, 2, 3]            | True           |
| `for i in a` | 迭代                 | for x in [1, 2]: print x, | 1 2            |

<font size="5">列表截取</font>

````py
L = ['spam', 'Spam', 'SPAM!']
````

| 表达式   | 结果              | 描述                               |
| :------- | :---------------- | :--------------------------------- |
| `L[2]`   | 'SPAM!'           | 读取第三个元素                     |
| `L[-2]`  | 'Spam'            | 读取倒数第二个元素                 |
| `L[1:]`  | ('Spam', 'SPAM!') | 从第二个元素开始截取元素,直到末尾  |
| `L[1:3]` | ('Spam', 'SPAM!') | 截取元素, 从第二个开始到第三个元素 |

<font size="5">列表函数&方法</font>

Python包含以下函数

| 函数                               | 描述                 |
| :--------------------------------- | :------------------- |
| [`cmp(list, list2)`](#1-cmp-方法)  | 比较两个元组元素     |
| [`len(list)`](#2-len-方法)         | 计算元组元素个数     |
| [`max(list)`](#3-max-方法min-方法) | 返回元组中元素最大值 |
| [`min(list)`](#3-max-方法min-方法) | 返回元组中元素最小值 |
| [`list(seq)`](#4-序列数据类型转换) | 将列表转换为元组     |

Python包含以下方法

| 方法                                           | 描述                                                             |
| :--------------------------------------------- | :--------------------------------------------------------------- |
| `list.append(obj)`                             | 在列表末尾添加新的对象                                           |
| `list.count(obj)`                              | 统计某个元素在列表中出现的次数                                   |
| `list.extend(seq)`                             | 在列表末尾一次性追加另一个序列中的多个值(用新列表扩展原来的列表) |
| `list.index(obj)`                              | 从列表中找出某个值第一个匹配项的索引位置                         |
| `list.insert(index, obj)`                      | 将对象插入列表                                                   |
| `list.pop([index=-1])`                         | 移除列表中的一个元素(默认最后一个元素), 并且返回该元素的值       |
| `list.remove(obj)`                             | 移除列表中某个值的第一个匹配项                                   |
| `list.reverse()`                               | 反向列表中元素                                                   |
| `list.sort(cmp=None, key=None, reverse=False)` | 对原列表进行排序                                                 |

关于列表的更多功能,参考[Numpy教程]()<span style="font-size: 10px;" class="hidden-text">(还没做)</span>

#### 2. 元组

<b>元组(tuple)</b>与列表类似, 不同之处在于元组的元素不能修改. 

只需要在小括号中添加元素, 并使用逗号隔开即可创建元组.
````py
tup1 = ()		# 创建一个空元组
tup2 = (1,)		# 当元组内只有1个元素时,  需要在元素后加逗号

# 元组可以包含字符串和数字
tup3 = ('physics', 'chemistry', 1997, 2000)
tup4 = (1, 2, 3, 4, 5 )

# 元组可以不加括号
tup5 = "a", "b", "c", "d"
tup6 = 1, 2, 3, 4, 5
````

元组与字符串类似, 下标索引从0开始, 可以进行截取, 组合等.

<font size="5">访问元组</font>

元组可以使用下标索引来访问元组中的值

````py
tup1 = ('physics', 'chemistry', 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7 )
 
print "tup1[0]: ", tup1[0]			# tup1[0]: physics
print "tup2[1:5]: ", tup2[1:5]		# tup2[1:5]: 2 3 4 5
````

<font size="5">修改元组</font>

元组中的元素值是不允许修改的, 但可以对元组进行连接组合

````py
 
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
 
# 以下修改元组元素操作是非法的. 
# tup1[0] = 100
 
# 创建一个新的元组
tup3 = tup1 + tup2
print tup3				# 输出: (12, 34.56, 'abc', 'xyz')
````

<font size="5">删除元组</font>

元组中的元素值是不允许删除的, 但可以使用`del`语句来删除整个元组

````py
tup = ('physics', 'chemistry', 1997, 2000)
 
print tup
del tup
print "After deleting tup : "
print tup		# 此处会报错
````

以上实例元组被删除后, 输出变量会有异常信息. 输出如下:

```
('physics', 'chemistry', 1997, 2000)
After deleting tup :
Traceback (most recent call last):
  File "test.py", line 9, in <module>
	print tup
NameError: name 'tup' is not defined
```

<font size="5">元组运算符</font>

与字符串一样, 元组之间可以使用 + 号和 * 号进行运算. 这就意味着他们可以组合和复制, 运算后会生成一个新的元组

| 代码           | 描述         | 示例                         | 结果           |
| :------------- | :----------- | :--------------------------- | :------------- |
| [`len`]        | 计算元素个数 | len((1, 2, 3))               | 3              |
| `+`            | 连接         | (1, 2) + (3, 4)              | (1, 2, 3, 4)   |
| `*`            | 复制         | ('Hi!',) * 2                 | ('Hi!', 'Hi!') |
| `in`           | 元素是否存在 | 3 in (1, 2, 3)               | True           |
| `for i in ...` | 迭代         | for x in (1, 2, 3): print x, | 1 2 3          |

<font size="5">元组索引, 截取</font>

因为元组也是一个序列, 所以我们可以访问元组中的指定位置的元素, 也可以截取索引中的一段元素

````py
L = ('spam', 'Spam', 'SPAM!')
````

| 表达式   | 结果              | 描述                               |
| :------- | :---------------- | :--------------------------------- |
| `L[2]`   | 'SPAM!'           | 读取第三个元素                     |
| `L[-2]`  | 'Spam'            | 反向读取,  读取倒数第二个元素      |
| `L[1:]`  | ('Spam', 'SPAM!') | 从第二个开始截取元素,直到末尾      |
| `L[1:3]` | ('Spam', 'SPAM!') | 截取元素, 从第二个开始到第三个元素 |

<font size="5">无关闭分隔符</font>

任意无符号的对象,  以逗号隔开, 默认为元组

````py
print 'abc', -4.24e93, 18+6.6j, 'xyz'
x, y = 1, 2
print "Value of x , y : ", x,y
````

运行结果为:

```
abc -4.24e+93 (18+6.6j) xyz
Value of x , y : 1 2
```

<font size="5">元组内置函数</font>

| 函数                                 | 描述                 |
| :----------------------------------- | :------------------- |
| [`cmp(tuple1, tuple2)`](#1-cmp-方法) | 比较两个元组元素     |
| [`len(tuple)`](#2-len-方法)          | 计算元组元素个数     |
| [`max(tuple)`](#3-max-方法min-方法)  | 返回元组中元素最大值 |
| [`min(tuple)`](#3-max-方法min-方法)  | 返回元组中元素最小值 |
| [`tuple(seq)`](#4-序列数据类型转换)) | 将列表转换为元组     |

### 2. 字典

>字典是一种可变容器模型, 且可存储任意类型对象

字典的每个键值 `key:value` 对用冒号 `:` 分割, 每个键值对之间用逗号 `,` 分割, 整个字典包括在花括号 `{}` 中

````py
diction1 = {key1 : value1, key2 : value2 }
````

❗`dict` 作为 Python 的关键字和内置函数, 变量名不建议命名为 `dict`

键一般是唯一的. 如果重复, 最后的一个键值对会替换前面的, 值不需要唯一

````py
>>> tinydict = {'a': 1, 'b': 2, 'b': '3'}
>>> tinydict['b']
'3'
>>> tinydict
{'a': 1, 'b': '3'}
````

值可以取任何数据类型, 但键必须是不可变的, 如字符串, 数字或元组

<font size="5">访问字典里的值</font>

把相应的键放入方括弧`[]`

````py
>>> tinydict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
>>> print "tinydict['Name']: ", tinydict['Name']
tinydict['Name']: Zara
````

<font size="5">修改字典</font>

修改字典的方法是增加新的键/值对, 修改或删除已有键/值对. 使用`.clear`清空字典, 使用`del`删除字典或字典内某个键

````py
tinydict = {'Name': 'Zara', 'Age': 7, 'Class': 'One'}
 
tinydict['Age'] = 8 # 更新
tinydict['School'] = "OX" # 添加

print "tinydict['Age']: ", tinydict['Age']
print "tinydict['School']: ", tinydict['School']
 
 del tinydict['Name']  # 删除键是'Name'的条目
 tinydict.clear()      # 清空字典所有条目
 del tinydict          # 删除字典
````

输出如下

```
tinydict['Age']:  8
tinydict['School']:  OX
```

<font size="5">字典键的特性</font>

字典值可以没有限制地取任何 python 对象, 既可以是标准的对象, 也可以是用户定义的, 但键不行

- 不允许同一个键出现两次
- 键必须不可变. 可以用数字, 字符串或元组充当, 不能用列表

<font size="5">字典内置函数&方法</font>

| 函数                               | 描述                                             |
| :--------------------------------- | :----------------------------------------------- |
| [`cmp(dict1, dict2)`](#1-cmp-方法) | 比较两个字典元素                                 |
| [`len(dict)`](#2-len-方法)         | 计算字典元素个数,  即键的总数                    |
| [`str(dict)`]                      | 输出字典可打印的字符串表示                       |
| [`type(variable)`]                 | 返回输入的变量类型, 如果变量是字典就返回字典类型 |

| 方法                                 | 描述                                                                                           |
| :----------------------------------- | :--------------------------------------------------------------------------------------------- |
| `dict.clear()`                       | 删除字典内所有元素                                                                             |
| `dict.copy()`                        | 返回一个字典的浅复制                                                                           |
| `dict.fromkeys(seq[, val])`          | 创建一个新字典, 以序列 seq 中元素做字典的键, val 为字典所有键对应的初始值                      |
| `dict.get(key, default=None)`        | 返回指定键的值, 如果值不在字典中返回default值                                                  |
| `dict.has_key(key)`                  | 如果键在字典dict里返回true, 否则返回false. <spna style="color: #e70000;">Python3 不支持</spna> |
| `dict.items()`                       | 以列表返回可遍历的 (键, 值) 元组数组                                                           |
| `dict.keys()`                        | 以列表返回一个字典所有的键                                                                     |
| `dict.setdefault(key, default=None)` | 和get()类似, 但如果键不存在于字典中, 将会添加键并将值设为default                               |
| `dict.update(dict2)`                 | 把字典dict2的键/值对更新到dict里                                                               |
| `dict.values()`                      | 以列表返回字典中的所有值                                                                       |
| `pop(key[,default])`                 | 删除字典给定键 key 所对应的值, 返回值为被删除的值. key值必须给出.  否则, 返回default值         |
| `popitem()`                          | 返回并删除字典中的最后一对键和值                                                               |

## 4. 面向对象编程

### 1. 对象和类

> 类是一个用于创建对象的 "蓝图" 或模板. 每个基于类的实例被称为对象

**格式**

````py
class Apple:  # 创建类以首字母大写与其它变量区分
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def colour(self):
        print(f"{self.name} is black")

# 创建Apple类的一个实例
First_Apple = Apple("Bad Apple", 3)
First_Apple()  # 输出: Bad Apple is black
````

`方法`: 在类内部定义的函数称为方法. 方法定义了类的行为. 例如 `colour`方法输出苹果颜色.

`__init__`方法的第一个参数永远是self, 表示创建的实例本身. 因此在`__init__`方法内部可以把各种属性绑定到self, 因为self就指向创建的实例本身. `__init__`方法会在创建类时自动执行.

`self` 参数代表类的一个实例(即对象), 其中的参数会被传递. 而`name`和`age`是传递给类的参数. 

### 数据封装

>数据封装即直接在类的内部定义访问数据的函数, 这样就把 "数据" 给封装起来了. 这些封装数据的函数是和类本身是关联起来的, 我们称之为类的方法

**例子**

````py
class Student(object):
	def __init__(self, name, score):
		self.name = name
		self.score = score

	def print_score(self):
		print('%s: %s' % (self.name, self.score))

bart = Student('Bart Simpson', 59)
````

要定义一个方法, 除了第一个参数是`self`外, 其他和普通函数一样.要调用一个方法, 只需要在实例变量上直接调用, 除了`self`不用传递, 其他参数正常传入

<font size="2" style="color:skyblue"><b>输入</b></font>

```
bart.print_score()
```

<font size="2" style="color:yellow"><b>输出</b></font>

```
Bart Simpson: 59
```

这样一来, 我们从外部看Student类, 就只需要知道, 创建实例需要给出`name`和`score`, 而如何打印, 都是在`Student`类的内部定义的, 这些数据和逻辑被 "封装" 起来了, 调用很容易, 但却不用知道内部实现的细节.

封装的另一个好处是可以给Student类增加新的方法


