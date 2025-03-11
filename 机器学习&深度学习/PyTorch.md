<font size=5>PyTorch指南</font>

<font size=4>目录</font>

- [1. 基本概念](#1-基本概念)
  - [1. `Tensor`张量](#1-tensor张量)
  - [2. `Autograd`自动求导](#2-autograd自动求导)
  - [3. `nn.Module`神经网络](#3-nnmodule神经网络)
  - [4. `Optimizers`优化器](#4-optimizers优化器)
  - [5. `Device`设备](#5-device设备)
- [2. 读取档案](#2-读取档案)
  - [1. `Dataset`](#1-dataset)
  - [2. `Dataloader`](#2-dataloader)


> PyTorch是一个机器学习库, 基于 Torch 库, 底层由C++实现. 它主要有两个用途: 
>
>- 类似Numpy但是能利用GPU加速
>- 基于带自动微分系统的深度神经网络

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

# 1. 基本概念

PyTorch 主要有以下几个基础概念: 张量(Tensor), 自动求导(Autograd), 神经网络模块(nn.Module), 优化器(optim)等

- `Tensor`张量: PyTorch 的核心数据结构, 支持多维数组, 并可以在 CPU 或 GPU 上进行加速计算. 
- `Autograd`自动求导: PyTorch 提供了自动求导功能, 可以轻松计算模型的梯度, 便于进行反向传播和优化. 
- `nn.Module`神经网络: PyTorch 提供了简单且强大的 API 来构建神经网络模型, 可以方便地进行前向传播和模型定义. 
- `Optimizers`优化器: 使用优化器(如 Adam, SGD 等)来更新模型的参数, 使得损失最小化. 
- `Device`设备: 可以将模型和张量移动到 GPU 上以加速计算. 

## 1. `Tensor`张量

`Tensor`张量是 张量是一个多维数组, 可以是标量, 向量, 矩阵或更高维度的数据结构

`Tensor`PyTorch 中的核心数据结构, 用于存储和操作多维数组. 

张量可以视为一个多维数组, 支持加速计算的操作. 

在 PyTorch 中, 张量的概念类似于 NumPy 中的数组, 但具有更强大的功能, 例如支持 GPU 加速和自动梯度计算

张量支持多种数据类型(整型, 浮点型, 布尔型等)。

张量可以存储在 CPU 或 GPU 中, GPU 张量可显著加速计算

- `Dimensionality`维度: 张量的维度指的是数据的多维数组结构. 例如, 一个标量(0维张量)是一个单独的数字, 一个向量(1维张量)是一个一维数组, 一个矩阵(2维张量)是一个二维数组, 以此类推. 

- `Shape`形状: 张量的形状是指每个维度上的大小. 例如, 一个形状为(3, 4)的张量意味着它有3行4列. 

- `Dtype`数据类型: 张量中的数据类型定义了存储每个元素所需的内存大小和解释方式. PyTorch支持多种数据类型, 包括整数型(如torch.int8, torch.int32), 浮点型(如torch.float32, torch.float64)和布尔型(torch.bool). 

**维度**

- 1D Tensor / Vector(一维张量/向量): 最基本的张量形式, 可以看作是一个数组

- 2D Tensor / Matrix(二维张量/矩阵): 二维数组, 通常用于表示矩阵

- 3D Tensor / Cube(三维张量/立方体): 三维数组, 可以看作是由多个矩阵堆叠而成的立方体

- 4D Tensor / Vector of Cubes(四维张量/立方体向量): 四维数组, 可以看作是由多个立方体组成的向量

- 5D Tensor / Matrix of Cubes(五维张量/立方体矩阵): 五维数组, 可以看作是由多个4D张量组成的矩阵, 可以理解为一个包含多个 4D 张量的集合。

**张量创建**

张量创建的方式如下:

| 方法                                | 说明                                                 | 示例                                      |
| :---------------------------------- | :--------------------------------------------------- | :---------------------------------------- |
| `torch.tensor(data)`                | 从 Python 列表或 NumPy 数组创建张量                  | x = torch.tensor([[1, 2], [3, 4]])        |
| `torch.zeros(size)`                 | 创建一个全为零的张量                                 | x = torch.zeros((2, 3))                   |
| `torch.ones(size)`                  | 创建一个全为 1 的张量                                | x = torch.ones((2, 3))                    |
| `torch.empty(size)`                 | 创建一个未初始化的张量                               | x = torch.empty((2, 3))                   |
| `torch.rand(size)`                  | 创建一个服从均匀分布的随机张量, 值在 [0, 1)          | x = torch.rand((2, 3))                    |
| `torch.randn(size)`                 | 创建一个服从正态分布的随机张量, 均值为 0, 标准差为 1 | x = torch.randn((2, 3))                   |
| `torch.arange(start, end, step)`    | 创建一个一维序列张量, 类似于 Python 的 range         | x = torch.arange(0, 10, 2)                |
| `torch.linspace(start, end, steps)` | 创建一个在指定范围内等间隔的序列张量                 | x = torch.linspace(0, 1, 5)               |
| `torch.eye(size)`                   | 创建一个单位矩阵(对角线为 1, 其他为 0)               | x = torch.eye(3)                          |
| `torch.from_numpy(ndarray)`         | 将 NumPy 数组转换为张量                              | x = torch.from_numpy(np.array([1, 2, 3])) |

````py
import torch

# 创建一个 2x3 的全 0 张量
a = torch.zeros(2, 3)
print(a)

# 创建一个 2x3 的全 1 张量
b = torch.ones(2, 3)
print(b)

# 创建一个 2x3 的随机数张量
c = torch.randn(2, 3)
print(c)

# 从 NumPy 数组创建张量
import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备(CPU/GPU)上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)
````

运行如下:

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[ 1.0189, -0.5718, -1.2814],
        [-0.5865,  1.0855,  1.1727]])
tensor([[1, 2],
        [3, 4]])
tensor([[-0.3360,  0.2203,  1.3463],
        [-0.5982, -0.2704,  0.5429]])
```

使用 `torch.tensor()` 函数, 可以将一个列表或数组转换为张量

````py
import torch

tensor = torch.tensor([1, 2, 3])
print(tensor)
````

运行如下

```
tensor([1, 2, 3])
```

使用 `torch.from_numpy()` 可以将 NumPy 数组转换为张量

````py
import numpy as np

np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(tensor)
````

输出如下

```
tensor([1, 2, 3])
```

**张量的属性**

| 属性               | 说明                           |
| :----------------- | :----------------------------- |
| `.shape`           | 获取张量的形状                 |
| `.size()`          | 获取张量的形状                 |
| `.dtype`           | 获取张量的数据类型             |
| `.device`          | 查看张量所在的设备 (CPU/GPU)   |
| `.dim()`           | 获取张量的维度数               |
| `.requires_grad`   | 是否启用梯度计算               |
| `.numel()`         | 获取张量中的元素总数           |
| `.is_cuda`         | 检查张量是否在 GPU 上          |
| `.T`               | 获取张量的转置(适用于 2D 张量) |
| `.item()`          | 获取单元素张量的值             |
| `.is_contiguous()` | 检查张量是否连续存储           |

````py
import torch

# 创建一个 2D 张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 张量的属性
print("Tensor:\n", tensor)
print("Shape:", tensor.shape)  # 获取形状
print("Size:", tensor.size())  # 获取形状(另一种方法)
print("Data Type:", tensor.dtype)  # 数据类型
print("Device:", tensor.device)  # 设备
print("Dimensions:", tensor.dim())  # 维度数
print("Total Elements:", tensor.numel())  # 元素总数
print("Requires Grad:", tensor.requires_grad)  # 是否启用梯度
print("Is CUDA:", tensor.is_cuda)  # 是否在 GPU 上
print("Is Contiguous:", tensor.is_contiguous())  # 是否连续存储

# 获取单元素值
single_value = torch.tensor(42)
print("Single Element Value:", single_value.item())

# 转置张量
tensor_T = tensor.T
print("Transposed Tensor:\n", tensor_T)
````

运行如下

```
Tensor:
 tensor([[1., 2., 3.],
         [4., 5., 6.]])
Shape: torch.Size([2, 3])
Size: torch.Size([2, 3])
Data Type: torch.float32
Device: cpu
Dimensions: 2
Total Elements: 6
Requires Grad: False
Is CUDA: False
Is Contiguous: True
Single Element Value: 42
Transposed Tensor:
 tensor([[1., 4.],
         [2., 5.],
         [3., 6.]])
```

**张量的操作**

*基础操作*
| 操作                    | 说明                         |
| :---------------------- | :--------------------------- |
| `+, -, *, /`            | 元素级加法, 减法, 乘法, 除法 |
| `torch.matmul(x, y)`    | 矩阵乘法                     |
| `torch.dot(x, y)`       | 向量点积(仅适用于 1D 张量)   |
| `torch.sum(x)`          | 求和                         |
| `torch.mean(x)`         | 求均值                       |
| `torch.max(x)`          | 求最大值                     |
| `torch.min(x)`          | 求最小值                     |
| `torch.argmax(x, dim)`  | 返回最大值的索引(指定维度)   |
| `torch.softmax(x, dim)` | 计算 softmax(指定维度)       |

*形状操作*
| 操作                     | 说明                       |
| :----------------------- | :------------------------- |
| `x.view(shape)`          | 改变张量的形状(不改变数据) |
| `x.reshape(shape)`       | 类似于 view, 但更灵活      |
| `x.t()`                  | 转置矩阵                   |
| `x.unsqueeze(dim)`       | 在指定维度添加一个维度     |
| `x.squeeze(dim)`         | 去掉指定维度为 1 的维度    |
| `torch.cat((x, y), dim)` | 按指定维度连接多个张量     |

````py
# 张量相加
e = torch.randn(2, 3)
f = torch.randn(2, 3)
print(e + f)

# 逐元素乘法
print(e * f)

# 张量的转置
g = torch.randn(3, 2)
print(g.t())  # 或者 g.transpose(0, 1)

# 张量的形状
print(g.shape)  # 返回形状
````

**张量与设备**

PyTorch 张量可以存在于不同的设备上, 包括CPU和GPU, 可以将张量移动到 GPU 上以加速计算

检查GPU是否可用

````py
torch.cuda.is_available()  # 返回 True 或 False
````

将张量移动到GPU

````py
if torch.cuda.is_available():
    tensor_gpu = tensor_from_list.to('cuda')  # 将张量移动到GPU
````

**梯度和自动微分**

PyTorch张量支持自动微分, 这是深度学习中的关键特性. 创建一个需要梯度的张量时, PyTorch可以自动计算其梯度

````py
# 创建一个需要梯度的张量
tensor_requires_grad = torch.tensor([1.0], requires_grad=True)

# 进行一些操作
tensor_result = tensor_requires_grad * 2

# 计算梯度
tensor_result.backward()
print(tensor_requires_grad.grad)  # 输出梯度
````

**内存和性能**

PyTorch 张量还提供了一些内存管理功能, 比如`.clone()`, `.detach()` 和 `.to()` 方法, 它们可以优化内存使用和提高性能

## 2. `Autograd`自动求导

自动求导(Automatic Differentiation, 简称Autograd)是深度学习框架中的一个核心特性, 它允许计算机自动计算数学函数的导数. 

在深度学习中, 自动求导主要用于两个方面: 一是在训练神经网络时计算梯度, 二是进行反向传播算法的实现. 

自动求导基于链式法则(Chain Rule), 这是一个用于计算复杂函数导数的数学法则. 链式法则表明, 复合函数的导数是其各个组成部分导数的乘积. 在深度学习中, 模型通常是由许多层组成的复杂函数, 自动求导能够高效地计算这些层的梯度. 

动态图与静态图: 

- `Dynamic Graph`动态图: 在动态图中, 计算图在运行时动态构建. 每次执行操作时, 计算图都会更新, 这使得调试和修改模型变得更加容易. PyTorch使用的是动态图. 

- `Static Graph`静态图: 在静态图中, 计算图在开始执行之前构建完成, 并且不会改变. TensorFlow最初使用的是静态图, 但后来也支持动态图

PyTorch 提供了自动求导功能, 通过 `autograd` 模块来自动计算梯度. 

`torch.Tensor` 对象有一个 `requires_grad` 属性, 用于指示是否需要计算该张量的梯度. 

当你创建一个 `requires_grad=True` 的张量时, PyTorch 会自动跟踪所有对它的操作, 以便在之后计算梯度. 

**创建需要梯度的张量**

````py
# 创建一个需要计算梯度的张量
x = torch.randn(2, 2, requires_grad=True)
print(x)

# 执行某些操作
y = x + 2
z = y * y * 3
out = z.mean()

print(out)
````

运行结果如下

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[ 1.0189, -0.5718, -1.2814],
        [-0.5865,  1.0855,  1.1727]])
tensor([[1, 2],
        [3, 4]])
tensor([[-0.3360,  0.2203,  1.3463],
        [-0.5982, -0.2704,  0.5429]])
tianqixin@Mac-mini runoob-test % python3 test.py
tensor([[-0.1908,  0.2811],
        [ 0.8068,  0.8002]], requires_grad=True)
tensor(18.1469, grad_fn=<MeanBackward0>)
```

**反向传播(Backpropagation)**

一旦定义了计算图, 可以通过 `.backward()` 方法来计算梯度. 

````py
# 反向传播, 计算梯度
out.backward()

# 查看 x 的梯度
print(x.grad)
````

在神经网络训练中, 自动求导主要用于实现反向传播算法. 

反向传播是一种通过计算损失函数关于网络参数的梯度来训练神经网络的方法. 在每次迭代中, 网络的前向传播会计算输出和损失, 然后反向传播会计算损失关于每个参数的梯度, 并使用这些梯度来更新参数. 

**停止梯度计算**

如果你不希望某些张量的梯度被计算(例如, 当你不需要反向传播时), 可以使用 `torch.no_grad()` 或设置 `requires_grad=False`

````py
# 使用 torch.no_grad() 禁用梯度计算
with torch.no_grad():
    y = x * 2
````

## 3. `nn.Module`神经网络

神经网络是一种模仿人脑神经元连接的计算模型, 由多层节点(神经元)组成, 用于学习数据之间的复杂模式和关系. 

神经网络通过调整神经元之间的连接权重来优化预测结果, 这一过程涉及前向传播, 损失计算, 反向传播和参数更新. 

神经网络的类型包括前馈神经网络, 卷积神经网络(CNN), 循环神经网络(RNN)和长短期记忆网络(LSTM), 它们在图像识别, 语音处理, 自然语言处理等多个领域都有广泛应用. 

PyTorch 提供了一个非常方便的接口来构建神经网络模型, 即 `torch.nn.Module`. 

我们可以继承 `nn.Module` 类并定义自己的网络层. 

**创建一个简单的神经网络**

````py
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)
        return x

# 创建网络实例
model = SimpleNN()

# 打印模型结构
print(model)
````

运行结果如下:

```
SimpleNN(
  (fc1): Linear(in_features=2, out_features=2, bias=True)
  (fc2): Linear(in_features=2, out_features=1, bias=True)
)
```

**训练过程:**

1. `Forward Propagation`前向传播: 在前向传播阶段, 输入数据通过网络层传递, 每层应用权重和激活函数, 直到产生输出. 

2. `Calculate Loss`计算损失: 根据网络的输出和真实标签, 计算损失函数的值. 

3. `Backpropagation`反向传播: 反向传播利用自动求导技术计算损失函数关于每个参数的梯度. 

4. `Parameter Update`参数更新: 使用优化器根据梯度更新网络的权重和偏置. 

5. `Iteration`迭代: 重复上述过程, 直到模型在训练数据上的性能达到满意的水平

**前向传播与损失计算**

````py
# 随机输入
x = torch.randn(1, 2)

# 前向传播
output = model(x)
print(output)

# 定义损失函数(例如均方误差 MSE)
criterion = nn.MSELoss()

# 假设目标值为 1
target = torch.randn(1, 1)

# 计算损失
loss = criterion(output, target)
print(loss)
````

## 4. `Optimizers`优化器

优化器在训练过程中更新神经网络的参数, 以减少损失函数的值。

PyTorch 提供了多种优化器, 例如 SGD, Adam 等。

**使用优化器进行参数更新:**

````py
# 定义优化器(使用 Adam 优化器)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练步骤
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 反向传播
optimizer.step()  # 更新参数
````

**训练模型**

训练模型是机器学习和深度学习中的核心过程, 旨在通过大量数据学习模型参数, 以便模型能够对新的, 未见过的数据做出准确的预测。

训练模型通常包括以下几个步骤:
1. 数据准备:
   1. 收集和处理数据, 包括清洗, 标准化和归一化
   2. 将数据分为训练集, 验证集和测试集
2. 定义模型:
   1. 选择模型架构, 例如决策树, 神经网络等
   2. 初始化模型参数(权重和偏置)
3. 选择损失函数:
   1. 根据任务类型(如分类, 回归)选择合适的损失函数
4. 选择优化器:
   1. 选择一个优化算法, 如SGD, Adam等, 来更新模型参数
5. 前向传播:
   1. 在每次迭代中, 将输入数据通过模型传递, 计算预测输出
6. 计算损失
   1. 使用损失函数评估预测输出与真实标签之间的差异
7. 反向传播
   1. 利用自动求导计算损失相对于模型参数的梯度
8. 参数更新
   1. 根据计算出的梯度和优化器的策略更新模型参数
9. 迭代优化
   1. 重复步骤5-8, 直到模型在验证集上的性能不再提升或达到预定的迭代次数
10. 评估和测试
    1. 使用测试集评估模型的最终性能, 确保模型没有过拟合
11. 模型调优
    1. 根据模型在测试集上的表现进行调参, 如改变学习率, 增加正则化等
12. 部署模型
    1. 将训练好的模型部署到生产环境中, 用于实际的预测任务

````py
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)
        return x

# 2. 创建模型实例
model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 假设我们有训练数据 X 和 Y
X = torch.randn(10, 2)  # 10 个样本, 2 个特征
Y = torch.randn(10, 1)  # 10 个目标值

# 5. 训练循环
for epoch in range(100):  # 训练 100 轮
    optimizer.zero_grad()  # 清空之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    # 每 10 轮输出一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
````

输出如下

```
Epoch [10/100], Loss: 1.7180
Epoch [20/100], Loss: 1.6352
Epoch [30/100], Loss: 1.5590
Epoch [40/100], Loss: 1.4896
Epoch [50/100], Loss: 1.4268
Epoch [60/100], Loss: 1.3704
Epoch [70/100], Loss: 1.3198
Epoch [80/100], Loss: 1.2747
Epoch [90/100], Loss: 1.2346
Epoch [100/100], Loss: 1.1991
```

在每 10 轮, 程序会输出当前的损失值, 帮助我们跟踪模型的训练进度。随着训练的进行, 损失值应该会逐渐降低, 表示模型在不断学习并优化其参数。

训练模型是一个迭代的过程, 需要不断地调整和优化, 直到达到满意的性能。这个过程涉及到大量的实验和调优, 目的是使模型在新的, 未见过的数据上也能有良好的泛化能力。

## 5. `Device`设备

PyTorch 允许你将模型和数据移动到 GPU 上进行加速。

使用 `torch.device` 来指定计算设备。

**将模型和数据移至 GPU:**

````py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到设备
model.to(device)

# 将数据移动到设备
X = X.to(device)
Y = Y.to(device)
````

在训练过程中, 所有张量和模型都应该移到同一个设备上(要么都在 CPU 上, 要么都在 GPU 上)

# 2. 读取档案

在PyTorch中,可以使用`Dataset`或`Dataloader`加载数据

通常 `Dataset` 用于定义数据集，而 `DataLoader` 用于加载数据

| 对比           | `Dataset`                 | `Dataloader`                                  |
| :------------- | :------------------------ | :-------------------------------------------- |
| 作用           | 负责数据的存储和索引访问  | 负责批量加载数据、打乱数据、多线程处理等      |
| 是否支持 batch | 不支持                    | 支持 batch 处理                               |
| 是否支持多线程 | 不支持                    | 通过 `num_workers` 参数支持                   |
| 是否打乱数据   | 不支持                    | 通过 `shuffle=True` 支持                      |
| 调用方式       | `dataset[i]` 获取单个样本 | 通过 `for batch in dataloader` 迭代获取 batch |

## 1. `Dataset`

`Dataset` 主要用于定义数据的读取方式，提供数据的索引访问功能，是 PyTorch 数据加载机制的基础

`Dataset`通常需要继承 `torch.utils.data.Dataset` 并实现：

- __len__(): 返回数据集的样本数量。
- __getitem__(index): 通过索引返回对应的样本（通常是数据和标签）

````py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
````

以上`Dataset`允许通过索引访问数据,例如:

````py
dataset = MyDataset([1, 2, 3, 4], [0, 1, 0, 1])
print(dataset[0])  # 输出: (1, 0)
````

## 2. `Dataloader`

`DataLoader` 是 PyTorch 提供的一个数据加载器，它的作用是：

- 自动化批量处理（batching）
- 打乱数据（shuffling）
- 使用多进程加载数据（num_workers）
- 自定义采样方式（通过 sampler）

````py
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print(batch)
````

这会自动将 `dataset` 中的数据分成 `batch`，每次返回 2 个样本，并且可以选择是否打乱数据