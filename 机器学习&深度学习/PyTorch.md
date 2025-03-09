<font size=5>PyTorch指南</font>

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

`Tensor`张量是 PyTorch 中的核心数据结构, 用于存储和操作多维数组. 

张量可以视为一个多维数组, 支持加速计算的操作. 

在 PyTorch 中, 张量的概念类似于 NumPy 中的数组, 但是 PyTorch 的张量可以运行在不同的设备上, 比如 CPU 和 GPU, 这使得它们非常适合于进行大规模并行计算, 特别是在深度学习领域. 

- `Dimensionality`维度：张量的维度指的是数据的多维数组结构. 例如, 一个标量(0维张量)是一个单独的数字, 一个向量(1维张量)是一个一维数组, 一个矩阵(2维张量)是一个二维数组, 以此类推. 

- `Shape`形状：张量的形状是指每个维度上的大小. 例如, 一个形状为(3, 4)的张量意味着它有3行4列. 

- `Dtype`数据类型：张量中的数据类型定义了存储每个元素所需的内存大小和解释方式. PyTorch支持多种数据类型, 包括整数型(如torch.int8, torch.int32), 浮点型(如torch.float32, torch.float64)和布尔型(torch.bool). 

**张量创建**

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

# 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)
````

运行如下:

````
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
````

**常用张量操作**

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


# 2. 读取档案

在PyTorch中,可以使用`Dataset`或`Dataloader`读取档案