<font size=5>Threading多线程</font>

<font size=5>目录</font>

- [1. `Thread` \& `Threading`](#1-thread--threading)
  - [1. `join`等待](#1-join等待)
  - [2. `Queue`](#2-queue)
  - [3. 锁定机制🔒](#3-锁定机制)
    - [1. `Lock` 互斥锁](#1-lock-互斥锁)
    - [2. `RLock` 可重入锁](#2-rlock-可重入锁)


# 1. `Thread` & `Threading`

Python通过两个标准库`thread`和`threading`提供对线程的支持.   
使用线程有两种方式: 函数或者用类来包装线程对象.
 
❗`thread` 模块并不是一个标准库中的模块, 它在 Python 3 中已经被弃用并且不再推荐使用.  
在 Python 3 中, 应该使用 `threading` 模块来处理多线程


<font size=5>线程模块</font>

`threading` 模块提供以下方法

- `threading.currentThread()` -- 返回当前的线程变量
- `threading.enumerate()` -- 返回一个包含正在运行的线程的list. 正在运行指线程启动后和结束前, 不包括启动前和终止后的线程
- `threading.activeCount()` -- 返回正在运行的线程数量, 与`len(threading.enumerate())`有相同的结果

除了使用方法外, 线程模块同样提供了`Thread`类来处理线程, `Thread`类提供了以下方法

- `run()` -- 用以表示线程活动的方法
- `start()` -- 启动线程活动
- `join([time])` -- 等待至线程中止. 这阻塞调用线程直至线程的`join()` 方法被调用中止(正常退出或者抛出未处理的异常)或者是可选的超时发生
- `isAlive()` -- 返回线程是否活动的
- `getName()` -- 返回线程名
- `setName()` -- 设置线程名

<font size=5>使用Threading模块创建线程</font>

````py
threading.Thread ( function, args[, kwargs] )
````

**参数**

- `function` -- 线程函数
- `args` -- 传递给线程函数的参数, 必须是个`tuple`类型
- `kwargs` -- 可选参数

使用`Threading`模块创建线程, 直接从`threading.Thread`继承, 然后重写`__init__`方法和`run`方法

````py
import threading
import time

def thread_job():
    print('T1 start\n')
    for i in range(10):
        time.sleep(1)                       # 让程序等待,括号()内数字单位为秒
        print(f'{10 - i} seconds left')
    print('T1 finish\n')

def main():
    print(threading.active_count())         # 输出当前线程数
    print(threading.enumerate())            # 输出当前线程名字
    print(threading.current_thread())       # 正在运行该程序的线程
    added_thread = threading.Thread(target=thread_job, name='T1')      # 添加线程, 目标是thread_job, 线程名为T1
    added_thread.start()                    # 执行线程

if __name__ == '__main__':                  # 当目前脚本是主程序(即直接执行当前脚本),则 __name__ 的值是 '__main__'
    main()
````



输出如下

```
1
[<_MainThread(MainThread, started 23380)>]
<_MainThread(MainThread, started 23380)>
T1 start

10 seconds left
9 seconds left
8 seconds left
7 seconds left
6 seconds left
5 seconds left
4 seconds left
3 seconds left
2 seconds left
1 seconds left
T1 finish

all done
```

## 1. `join`等待

使用threading模块中的`join`方法可以延迟程序的执行

````py
added_thread.join()
````

当`added_thread`执行完后才会继续执行当前线程

以上一个示例为例

````py
import threading
import time

def thread_job():
    print('T1 start\n')
    print(threading.current_thread())
    for i in range(10):
        time.sleep(1)
        print(f'{10 - i} seconds left')
    print('T1 finish\n')

def main():
    print(threading.active_count())
    print(threading.enumerate())
    print(threading.current_thread())
    added_thread = threading.Thread(target=thread_job, name='T1')
    added_thread.start()
    added_thread.join()         # 等待added_thread结束
    print(threading.current_thread())
    print('all done')

if __name__ == '__main__':
    main()
````

输出如下

```
1
[<_MainThread(MainThread, started 25924)>]
<_MainThread(MainThread, started 25924)>
T1 start

<Thread(T1, started 9384)>
10 seconds left
9 seconds left
8 seconds left
7 seconds left
6 seconds left
5 seconds left
4 seconds left
3 seconds left
2 seconds left
1 seconds left
T1 finish

<_MainThread(MainThread, started 25924)>
all done
```

## 2. `Queue` 

`Queue`在`queue`模块中

Queue 可以

````py
import threading
import time
from queue import Queue

# 处理每个子列表的函数
def job(l):
    # 遍历每个元素, 将其平方
    for i in range(len(l)):
        l[i] = l[i] ** 2
    # 结果放入队列中
    q.put(l)

# 多线程处理的主函数
def multithreading(data):
    q = Queue()  # 创建队列用于存储线程处理的结果
    threads = []  # 存储线程对象的列表
    
    # 这里是原始数据, 每个子列表的数据会被平方
    # 注意: data 的初始化不应放在这里, 而是作为参数传入
    for i in range(3):  # 遍历每个子列表
        # 创建线程, 每个线程处理 data[i] 子列表
        t = threading.Thread(target=job, args=(data[i],))  # args 应该是元组, 数据传入时要加逗号
        t.start()  # 启动线程
        threads.append(t)  # 将线程加入线程列表
    
    # 等待所有线程完成
    for t in threads:
        t.join()  # 这里是拼写错误, 应该是 `t.join()`, 而不是 `thread.join()`

    # 从队列中获取处理后的数据
    results = []  # 存储结果的列表
    
    for k in range(3):  # 3个子列表的结果
        results.append(q.get())  # 从队列中取出每个线程的结果
    print(results)  # 输出所有处理后的结果

# 主程序入口
if __name__ == '__main__':
    # 主程序调用 multithreading 函数并传入数据
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 原始数据
    multithreading(data)
````

**分析**

1. `job(l)`：
   - 这个函数接受一个列表 `l`, 将列表中的每个元素平方. 
   - 然后, 处理完的数据被放入队列 `q` 中. 

2. `multithreading(data)`：

    - 这个函数接受一个数据 data(二维列表). 
    - 创建一个队列 q 用于线程间的结果传递. 
   - 为每个子列表创建一个线程, 线程会调用 job() 函数来处理子列表. 
  
3. 队列 `Queue()`：
    - `Queue` 是一个线程安全的队列, 用于在线程间传递数据. 
    - 每个线程将处理后的数据放入队列中, 主线程会从队列中获取数据. 

4. `threads.append(t)`：
    - `threads` 列表用于存储线程对象. 每次创建线程时, 都会将线程添加到这个列表中. 

**流程**

1. 初始化 `data` 为三组列表(每组含有三个数字).  
2. 为每组数据创建一个线程, 线程将数据中的数字平方.  
3. 每个线程将处理后的数据放入队列.  
4. 主线程等待所有子线程完成工作后, 从队列中取出处理结果并存储到   `results  中.  
5. 最终输出处理后的结果.  

输出如下

```
[[1, 4, 9], [16, 25, 36], [49, 64, 81]]
```
## 3. 锁定机制🔒

锁定机制是用于控制多线程访问共享资源的一种同步手段. 锁定可以防止多个线程同时修改同一数据, 从而避免竞态条件的发生  
Python的`threading`模块提供了多种锁定机制, 包括互斥锁`Mutex`和可重入锁`RLock`


### 1. `Lock` 互斥锁

`Lock`是`threading`中的一个方法  
Lock<span class='hidden-text'>指令锁</span>是可用的最低级的同步指令. Lock处于锁定状态时, 不被特定的线程拥有. Lock包含两种状态--锁定和非锁定, 以及两个基本的方法

**构造和方法**

````py
lock = threading.Lock()
lock.acquire([timeout])
lock.release()
````

`acquire` -- 使线程进入同步阻塞状态, 尝试获得锁定
`release()` -- 释放锁. 使用前线程必须已获得锁定, 否则将抛出异常

未使用锁时

````py
import threading
import time

num = 0

def show(arg):
    global num
    time.sleep(1)
    num +=1
    print('bb :{}'.format(num))

for i in range(5):
    t = threading.Thread(target=show, args=(i,))  # 注意传入参数后一定要有【, 】逗号
    t.start()

print('main thread stop')
````

输出如下

```
main thread stop
bb :1
bb :2
bb :3bb :4
bb :5
```

使用锁时

````py
import threading
import time

num = 0

lock = threading.RLock()


# 调用acquire([timeout])时, 线程将一直阻塞, 
# 直到获得锁定或者直到timeout秒后(timeout参数可选). 
# 返回是否获得锁. 
def Func():
    lock.acquire()
    global num
    num += 1
    time.sleep(1)
    print(num)
    lock.release()


for i in range(10):
    t = threading.Thread(target=Func)
    t.start()
````

输出如下

```
1
2
3
4
5
6
7
8
9
10
```

可以看出, 全局变量在在每次被调用时都要获得锁, 才能操作, 因此保证了共享数据的安全性

对于`Lock`对象而言, 如果一个线程连续两次`release`, 使得线程死锁. 所以Lock不常用, 一般采用`Rlock`进行线程锁的设定

### 2. `RLock` 可重入锁

`RLock`是可以被同一个线程请求多次的同步指令.  
`RLock`使用了"拥有的线程"和"递归等级"的概念, 处于锁定状态时, `RLock`被某个线程拥有.  
拥有`RLock`的线程可以再次调用`acquire()`, **释放锁时需要调用`release()`相同次数**.  
可以认为`RLock`包含一个锁定池和一个初始值为`0`的计数器, 每次成功调用 `acquire()`/`release()`, 计数器将`+1`/`-1`, 为`0`时锁处于未锁定状态

**构造和方法**

````py
lock = threading.RLock()
lock.acquire([timeout])
lock.release()
````

使用方法和`Lock`基本一样,参见[`Lock`](#1-lock-互斥锁)