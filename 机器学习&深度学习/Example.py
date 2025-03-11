'''
安装必要的库
'''
# 数值运算
import math
import numpy as np

# 读写数据
import pandas as pd
import os
import csv

# 进度条
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 绘制曲线
from torch.utils.tensorboard import SummaryWriter

'''
Utility Functions
'''

def same_seed(seed): 
    '''
    修正随机数生成器种子, 以提高可重复性
    : seed: 用于生成随机数的种子
    '''
    torch.backends.cudnn.deterministic = True       # 强制 PyTorch 的 CuDNN 后端使用确定性算法, 避免因算法选择带来的随机性
    torch.backends.cudnn.benchmark = False          # 禁用 CuDNN 的基准测试模式基准测试模式会根据硬件自动选择最优算法, 但可能导致结果不可重复；禁用后确保一致性
    np.random.seed(seed)                            # 设置 NumPy 的随机数生成器种子, 以确保每一轮生成的随机数在种子不变的情况下相同
    torch.manual_seed(seed)                         # 设置 PyTorch 在 CPU 上的随机数生成器种子, 确保 CPU 上的随机数生成可重复
    if torch.cuda.is_available():                   # 检查当前系统是否支持 CUDA
        torch.cuda.manual_seed_all(seed)            # 为所有 GPU 设置随机种子, 确保 GPU 上的随机数生成也具有可重复性
'''该函数通过设置 NumPy 和 PyTorch(包括 CPU 和 GPU)的随机种子, 以及调整 CuDNN 的行为, 来保证随机数生成的可重复性'''


def train_valid_split(data_set, valid_ratio, seed):
    '''
    将提供的训练数据分成训练集和验证集
    : data_set: 输入的数据集, 通常是一个 PyTorch 的 Dataset 对象或其他类似的可索引对象
    : valid_ratio: 验证集占总数据集的比例(浮点数, 范围在 0 到 1 之间)
    : seed: 随机种子, 用于控制数据划分的随机性
    '''
    valid_set_size = int(valid_ratio * len(data_set))           # 获取数据集的总长度(样本数量)并计算验证集的样本数, 将计算结果取整并赋给valid_set_size
    train_set_size = len(data_set) - valid_set_size             # 剩余的样本数分配给训练集, 确保训练集和验证集之和等于原始数据集大小
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    '''
    : random_split: 将数据集随机分割成指定大小的子集
    : [train_set_size, valid_set_size]: 指定两个子集的大小
    : generator=torch.Generator().manual_seed(seed): 创建一个随机数生成器, 并用 seed 初始化, 确保每次运行时划分结果一致(可重复性)
    : train_set: 子集, 是 PyTorch 的 Subset 对象, 代表训练集
    : valid_set: 同上, 是验证集
    '''
    return np.array(train_set), np.array(valid_set)             #  PyTorch 的子集对象转换为 NumPy 数组
'''该函数将输入的数据集按指定比例(alid_ratio)随机划分为训练集和验证集, 并通过种子(seed)确保划分的可重复性, 最后返回 NumPy 数组格式的结果'''

def predict(test_loader, model, device):
    '''
    : test_loader: 测试数据的加载器, 是 PyTorch 的 DataLoader 对象, 用于批量加载测试数据
    : model: 训练好的模型, 是 PyTorch 的 nn.Module 子类
    : device: 计算设备, 指定模型和数据运行的硬件
    '''
    model.eval()                                # 将模型设置为评估模式(在评估模式下, 模型会禁用 dropout 和批量归一化(BatchNorm)等训练时特有的行为, 确保预测结果一致且不受随机性影响)
    preds = []                                  # 用于存储每个批次的预测结果
    for x in tqdm(test_loader):                 # 使用tqdm显示进度条          
        x = x.to(device)                        # 将当前批次的数据 x 移动到指定的设备(CPU 或 GPU), 与模型的运行设备保持一致
        with torch.no_grad():                   # 禁用梯度计算, 减少内存使用并加速推理(预测时不需要反向传播)     
            pred = model(x)                     # 将输入 x 传入模型, 得到预测结果 pred(通常是一个张量, 表示模型输出, 如分类分数或回归值)
            preds.append(pred.detach().cpu())   # 从计算图中分离预测结果, 避免保留梯度信息, 并将预测结果移回 CPU, 方便后续处理(即使模型在 GPU 上运行). 最后将每个批次的预测结果添加到列表中
    preds = torch.cat(preds, dim=0).numpy()     # 将所有批次的预测结果(列表中的张量)沿着第 0 维(通常是批次维度)拼接成一个大张量, 转换为NumPy数组并赋值给preds
    return preds
'''该函数使用训练好的模型对测试数据进行预测, 逐批处理数据, 最终返回所有预测结果的 NumPy 数组'''

'''
Dataset
'''

class TheDataset(Dataset):                          # 继承自 torch.utils.data.Dataset, 用于数据加载的标准基类
    '''
    : x: 特征数据
    : y: 目标数据, 如果没有, 则用于预测场景
    '''
    def __init__(self, x, y=None):                  # 构造函数
        if y is None:                               # 如果 y 是 None(即没有目标数据), 直接将 self.y 设为 None, 表示这是一个无标签数据集(例如测试或预测场景)
            self.y = y
        else:                                       # 如果 y 存在, 将其转换为 PyTorch 的 FloatTensor(浮点张量), 存储在 self.y 中
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)               # 将 x 转换为 FloatTensor, 存储在 self.x 中
    '''初始化数据集, 将输入数据转换为 PyTorch 张量, 准备后续通过 DataLoader 加载'''

    def __getitem__(self, idx):                     # 索引方法
        '''
        : idx: 数据集的索引(整数)
        '''
        if self.y is None:                          # 如果 self.y 是 None(无标签), 返回对应索引的特征 self.x[idx]
            return self.x[idx]
        else:                                       # 如果 self.y 存在(有标签), 返回一个元组 (self.x[idx], self.y[idx]), 包含特征和目标
            return self.x[idx], self.y[idx]
    '''定义如何通过索引访问数据集中的单个样本支持两种模式: 有标签时返回 (x, y), 无标签时只返回 x '''

    def __len__(self):                              # 长度方法
        return len(self.x)                          # 返回特征数据 self.x 的长度
    '''告诉 PyTorch 数据集包含多少样本, 用于迭代或分割'''
'''TheDataset 是一个 PyTorch 数据集类, 支持有标签(训练/验证)和无标签(测试/预测)两种场景特征和目标数据会被转换为浮点张量'''

'''
神经网络模型
'''

class My_Model(nn.Module):                      # 继承自 torch.nn.Module, 定义神经网络的标准基类
    def __init__(self, input_dim):
        '''
        : input_dim: 输入特征的维度(整数), 由外部传入, 取决于数据集的特征数量'''
        super(My_Model, self).__init__()        # 调用父类 nn.Module 的构造函数, 初始化模型
        # TODO: 修改模型结构, 注意尺寸
        self.layers = nn.Sequential(            # 使用 nn.Sequential 定义一个顺序层结构, 以下是包含的层
            nn.Linear(input_dim, 16),           # 线性层, 将输入维度从 input_dim 映射到 16
            nn.ReLU(),                          # ReLU 激活函数, 引入非线性
            nn.Linear(16, 8),                   # 线性层, 将 16 维特征映射到 8 维
            nn.ReLU(),                          # 再次使用 ReLU 激活
            nn.Linear(8, 1)                     # 线性层, 将 8 维特征映射到 1 维输出
        )

    def forward(self, x):                       # 前向传播
        '''
        : x: 输入张量, 形状通常是 (B, input_dim), 其中 B 是批次大小
        '''
        x = self.layers(x)                      # 将输入 x 通过 self.layers 定义的层结构, 得到输出张量, 形状为 (B, 1)
        x = x.squeeze(1)                        # 移除张量的第 1 维(维度大小为 1) , 将 (B, 1) 变为 (B)
        return x
'''My_Model 是一个简单的全连接神经网络, 输入维度为 input_dim, 输出一个标量值(每个样本一个输出)'''
    
'''
选择特征
'''

def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    选择有用的特征进行回归
    : train_data: 训练数据, 通常是一个二维数组
    : valid_data: 验证数据, 格式与训练数据一致
    : test_data: 测试数据, 可能没有目标列
    : select_all: 布尔值, 默认为 True, 控制是否选择所有特征
    '''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]               # 从训练数据和验证数据中提取最后一列作为目标变量 y_train 和 y_valid
    raw_x_train, raw_x_valid = train_data[:,:-1], valid_data[:,:-1]     # 从训练数据和验证数据中提取除最后一列外的所有列, 作为特征 raw_x_train 和 raw_x_valid
    raw_x_test = test_data                                              # 测试数据直接赋值给 raw_x_test, 假设其没有目标列(仅特征)

    if select_all:                                                      # select_all 为 True 时
        feat_idx = list(range(raw_x_train.shape[1]))                    # 获取训练数据的特征数量(列数)并生成所有特征的索引列表
    else:                                                               
        feat_idx = [0,1,2,3,4] # TODO: 选择合适的特征列                  # 使用硬编码的特征索引
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
'''从训练, 验证和测试数据中提取特征和目标, 支持选择所有特征或指定特征子集'''
     
'''
训练循环
'''

def trainer(train_loader, valid_loader, model, config, device):
    '''
    : train_loader: 训练数据的 DataLoader, 提供批量训练数据
    : valid_loader: 验证数据的 DataLoader, 用于评估模型
    : model: 待训练的模型(如 My_Model 的实例)
    : config: 配置字典, 包含超参数(如学习率, 轮数等)
    : device: 计算设备(torch.device, 如 'cpu' 或 'cuda')
    '''

    # 定义损失函数
    criterion = nn.MSELoss(reduction='mean')                                                        # 使用均方误差(MSE)损失, reduction='mean' 表示对所有样本的损失取平均. 与 My_Model 的标量输出和回归任务一致

    # 定义优化算法
    # TODO: 查看 https://pytorch.org/docs/stable/optim.html 可获得更多可用算法
    # TODO: L2 正则化(优化器(权重衰减......))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)       # 使用随机梯度下降(SGD)优化器
    '''
    : model.parameters(): 模型的可训练参数
    : lr=config['learning_rate']: 学习率, 从配置中获取
    : momentum=0.9: 动量因子, 加速梯度下降
    '''

    # 初始化 TensorBoard 和模型保存目录
    writer = SummaryWriter()                                                                        # 绘制tensoboard, 用于可视化训练和验证损
    if not os.path.isdir('./models'):
        os.mkdir('./models')                                                                        # 创建保存模型权重的目录

    # 初始化训练参数
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    '''
    : n_epochs: 训练轮数, 从 config 获取
    : best_loss: 最佳验证损失, 初始为无穷大
    : step: 全局步数, 记录优化步数
    : early_stop_count: 早停计数器, 跟踪模型未改进的轮次
    '''

    # 训练循环
    for epoch in range(n_epochs):
        model.train()                                                                               # 将模型设置为训练模式(激活 dropout, BatchNorm 等)
        loss_record = []                                                                            # 记录每个批次的损失
        train_pbar = tqdm(train_loader, position=0, leave=True)                                     # 为训练数据添加进度条, 显示训练进度

        # 训练单个批次
        for x, y in train_pbar:
            optimizer.zero_grad()                                                                   # 清零梯度, 避免累积
            x, y = x.to(device), y.to(device)                                                       # 将数据移至设备, 与模型一致
            pred = model(x)                                                                         # 前向传播, 调用 My_Model.forward
            loss = criterion(pred, y)                                                               # 计算 MSE 损失
            loss.backward()                                                                         # 反向传播, 计算梯度
            optimizer.step()                                                                        # 更新模型参数
            step += 1
            loss_record.append(loss.detach().item())                                                # 记录损失(detach().item() 转为标量)
            
            # 进度条更新
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')                             # 显示当前 epoch
            train_pbar.set_postfix({'loss': loss.detach().item()})                                  # 显示最近批次的损失

        # 计算平均训练损失并记录
        mean_train_loss = sum(loss_record)/len(loss_record)                                         # 计算整个 epoch 的平均训练损失
        writer.add_scalar('Loss/train', mean_train_loss, step)                                      # 用 TensorBoard 记录

        # 验证阶段
        model.eval()                                                                                # 将模型切换到评估模式
        loss_record = []                                                                            # 初始化损失记录列表
        for x, y in valid_loader:                                                                   # 遍历验证数据加载器
            x, y = x.to(device), y.to(device)                                                       # 将数据移到指定设备
            with torch.no_grad():                                                                   # 禁用梯度计算, 节省内存
                pred = model(x)                                                                     # 模型前向传播, 将输入 x 传入模型, 得到预测结果 pred
                loss = criterion(pred, y)                                                           # 使用定义的损失函数(nn.MSELoss)计算预测值 pred 和真实值 y 之间的损失

            loss_record.append(loss.item())                                                         # 将当前批次的损失值添加到 loss_record 列表

        # 计算平均验证损失并打印
        mean_valid_loss = sum(loss_record)/len(loss_record)                                         # 计算验证集平均损失
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')        # 打印训练和验证损失
        writer.add_scalar('Loss/valid', mean_valid_loss, step)                                      # 记录到 TensorBoard

        # 保存最佳模型和早停机制
        if mean_valid_loss < best_loss:                                                             # 比较损失与最佳损失
            best_loss = mean_valid_loss                                                             # 记录小损失
            torch.save(model.state_dict(), config['save_path'])                                     # 保存模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0                                                                    # 重置早停计数器
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:                                                # 计数器达到设定上限
            print('\nModel is not improving, so we halt the training session.')
            return                                                                                  # 提前终止训练
'''训练模型, 优化参数, 保存最佳模型, 支持早停和损失可视化'''

'''
配置
config包含用于训练的超参数和保存模型的路径
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 优先使用GPU,  若没有, 使用CPU
config = {
    'seed': 1,                                              # 随机种子, 用于控制随机过程的可重复性. 可随意设置
    'select_all': True,                                     # 是否选择所有特征(布尔值)
    'valid_ratio': 0.2,                                     # 验证集占训练数据的比例
    'n_epochs': 3000,                                       # epochs数量 (训练的总轮数)          
    'batch_size': 256,                                      # batch大小 (每个批次的数据样本数)
    'learning_rate': 1e-5,                                  # 优化器的学习率 (较小的学习率表示参数更新步幅较小)
    'early_stop': 400,                                      # 早停的耐心值(patience), 即模型在多少轮未改进后停止训练   
    'save_path': './models/model.ckpt'                      # 保存模型的路径
}
''''程序的全局配置'''

'''
Dataloader
从文件中读取数据并设置训练集, 验证集和测试集
'''


same_seed(config['seed'])                                                                                               # 设置随机种子, 以实现结果可重复

train_data, test_data = pd.read_csv('./train.csv').values, pd.read_csv('./test.csv').values                 # 从 CSV 文件加载训练和测试数据, 并转为 NumPy 数组
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])                           # 将数据集分割为训练集和验证集

# 打印数据大小
print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")                                                                                   # 输出训练、验证和测试数据的大小, 验证分割是否正确

# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])       # 从数据中提取特征和目标
'''
: x_train: 训练特征
: x_valid: 验证特征
: x_test: 测试特征
: y_train: 训练目标
: y_valid: 验证目标
: train_data: 从 train_valid_split 返回的训练数据
: valid_data: 从 train_valid_split 返回的验证数据
: test_data: 从 CSV 加载的测试数据
: config['select_all'] = True: 选择所有特征
'''

print(f'number of features: {x_train.shape[1]}')                                                                        # 打印选择的特征数量

# 创建数据集, 将数据封装为 PyTorch 数据集
train_dataset, valid_dataset, test_dataset = TheDataset(x_train, y_train), \
                                            TheDataset(x_valid, y_valid), \
                                            TheDataset(x_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)                # 训练数据加载器, 迭代返回
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)                # 验证数据加载器
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)                 # 测试数据加载器
'''加载 COVID-19 数据, 分割训练/验证集, 选择特征, 创建 PyTorch 数据集和加载器, 为训练和预测做准备'''    

'''
开始训练
'''

model = My_Model(input_dim=x_train.shape[1]).to(device)             # 初始化模型, 移到 device, 准备处理 117 个特征的输入
trainer(train_loader, valid_loader, model, config, device)          # 调用 trainer 函数, 使用训练和验证数据加载器训练模型, 并根据配置优化参数

'''
测试
将模型对测试集的预测结果存储在 pred.csv 中
'''

def save_pred(preds, file):                                     # 将预测结果 preds 保存到指定的 CSV 文件中，格式为两列: id 和 tested_positive
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)         # 创建新的模型实例并移到指定设备
model.load_state_dict(torch.load(config['save_path']))          # 加载训练阶段保存的最佳模型参数
preds = predict(test_loader, model, device)                     # 使用加载的模型对测试数据进行预测
save_pred(preds, 'pred.csv')                                    # 将预测结果保存到 pred.csv 文件

'''
整体设计思路:
1. 数据驱动: 从 CSV 文件加载数据，预处理后分为训练、验证和测试集，利用特征预测目标(阳性率)。
2. 模型训练与评估: 设计一个简单的全连接神经网络，通过训练和验证优化模型参数，使用早停机制防止过拟合。
3. 可重复性与灵活性: 通过随机种子和配置字典控制实验一致性和超参数调整。
4. 结果输出: 对测试集进行预测并保存为指定格式的 CSV 文件，便于提交或后续分析。

程序遵循机器学习的标准pipeline: 
数据准备 -> 模型定义 -> 训练与验证 -> 预测 -> 结果保存
'''

'''
功能模块分解:
1. 随机种子设置(same_seed)
    目的: 确保随机过程(如数据分割、模型初始化)的可重复性。
    实现: 设置 NumPy 和 PyTorch 的随机种子，调整 CuDNN 行为。

2. 数据分割(train_valid_split)
    目的: 将训练数据分为训练集和验证集，用于模型优化和评估。
    实现: 按比例随机分割，使用种子控制一致性。

3. 特征选择(select_feat)
    目的: 从原始数据中提取特征和目标，支持灵活选择特征子集。
    实现: 分离特征和目标列，默认使用所有特征。

4. 数据集定义(TheDataset)
    目的: 将 NumPy 数据转换为 PyTorch 张量，支持训练和预测两种模式。
    实现: 继承 Dataset，定义数据访问接口。

5. 模型定义(My_Model)
    目的: 构建神经网络，映射特征到目标值。
    实现: 全连接网络(117 -> 16 -> 8 -> 1)，适合回归任务。

6. 训练逻辑(trainer)
    目的: 优化模型参数，保存最佳模型。
    实现: 使用 SGD 优化器和 MSE 损失，包含早停和 TensorBoard 可视化。

7. 预测逻辑(predict)
    目的: 对测试数据生成预测结果。
    实现: 批量推理，转为 NumPy 数组。

8. 结果保存(save_pred)
    目的: 将预测结果写入 CSV 文件。
    实现: 按指定格式保存 ID 和预测值。

9. 主流程
    目的: 协调各模块，完成数据处理、训练和预测。
    实现: 加载数据、初始化模型、调用训练和预测函数。
'''