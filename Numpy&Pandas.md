<font size=5>Python Numpy 和 Pandas笔记</font>
# 1.目录

- [1.目录](#1目录)
- [2.`Numpy`](#2numpy)
  - [1.`.array`](#1array)


# 2.`Numpy`

`Numpy`是以C语言编写的数据处理模块

## 1.`.array`

array方法将列表转为矩阵

````py
import numpy as np

array = np.array([[1, 2, 3],
                [4, 5, 6]])

print(array)
print('number of dim: ', array.ndim)
print('shape: ', array.shape)
print('size: ', array.size)
````

运行结果如下

```
[[1 2 3]
 [4 5 6]]
number of dim:  2
shape:  (2, 3)
size:  6
```