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
    torch.backends.cudnn.benchmark = False          # 禁用 CuDNN 的基准测试模式。基准测试模式会根据硬件自动选择最优算法, 但可能导致结果不可重复；禁用后确保一致性
    np.random.seed(seed)                            # 设置 NumPy 的随机数生成器种子, 以确保每一轮生成的随机数在种子不变的情况下相同
    torch.manual_seed(seed)                         # 设置 PyTorch 在 CPU 上的随机数生成器种子, 确保 CPU 上的随机数生成可重复
    if torch.cuda.is_available():                   # 检查当前系统是否支持 CUDA
        torch.cuda.manual_seed_all(seed)            # 为所有 GPU 设置随机种子, 确保 GPU 上的随机数生成也具有可重复性
'''该函数通过设置 NumPy 和 PyTorch(包括 CPU 和 GPU)的随机种子, 以及调整 CuDNN 的行为, 来保证随机数生成的可重复性'''


def train_valid_split(data_set, valid_ratio, seed):
    '''
    将提供的训练数据分成训练集和验证集
    : data_set: 输入的数据集, 通常是一个 PyTorch 的 Dataset 对象或其他类似的可索引对象。
    : valid_ratio: 验证集占总数据集的比例(浮点数, 范围在 0 到 1 之间)。
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
    model.eval()                                # 将模型设置为评估模式(在评估模式下, 模型会禁用 dropout 和批量归一化（BatchNorm）等训练时特有的行为, 确保预测结果一致且不受随机性影响)
    preds = []                                  # 用于存储每个批次的预测结果
    for x in tqdm(test_loader):                 # 使用tqdm显示进度条          
        x = x.to(device)                        # 将当前批次的数据 x 移动到指定的设备（CPU 或 GPU）, 与模型的运行设备保持一致
        with torch.no_grad():                   # 禁用梯度计算, 减少内存使用并加速推理（预测时不需要反向传播）     
            pred = model(x)                     # 将输入 x 传入模型, 得到预测结果 pred（通常是一个张量, 表示模型输出, 如分类分数或回归值）
            preds.append(pred.detach().cpu())   # 从计算图中分离预测结果, 避免保留梯度信息, 并将预测结果移回 CPU, 方便后续处理（即使模型在 GPU 上运行）. 最后将每个批次的预测结果添加到列表中
    preds = torch.cat(preds, dim=0).numpy()     # 将所有批次的预测结果（列表中的张量）沿着第 0 维（通常是批次维度）拼接成一个大张量, 转换为NumPy数组并赋值给preds
    return preds
'''该函数使用训练好的模型对测试数据进行预测，逐批处理数据，最终返回所有预测结果的 NumPy 数组'''

'''
Dataset
'''

class TheDataset(Dataset):                          # 继承自 torch.utils.data.Dataset，用于数据加载的标准基类
    '''
    : x: 特征数据
    : y: 目标数据, 如果没有, 则用于预测场景
    '''
    def __init__(self, x, y=None):                  # 构造函数
        if y is None:                               # 如果 y 是 None（即没有目标数据），直接将 self.y 设为 None，表示这是一个无标签数据集（例如测试或预测场景）
            self.y = y
        else:                                       # 如果 y 存在，将其转换为 PyTorch 的 FloatTensor（浮点张量），存储在 self.y 中
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)               # 将 x 转换为 FloatTensor，存储在 self.x 中
    '''初始化数据集，将输入数据转换为 PyTorch 张量，准备后续通过 DataLoader 加载'''

    def __getitem__(self, idx):                     # 索引方法
        '''
        : idx: 数据集的索引（整数）
        '''
        if self.y is None:                          # 如果 self.y 是 None（无标签），返回对应索引的特征 self.x[idx]
            return self.x[idx]
        else:                                       # 如果 self.y 存在（有标签），返回一个元组 (self.x[idx], self.y[idx])，包含特征和目标
            return self.x[idx], self.y[idx]
    '''定义如何通过索引访问数据集中的单个样本。支持两种模式：有标签时返回 (x, y)，无标签时只返回 x '''

    def __len__(self):                              # 长度方法
        return len(self.x)                          # 返回特征数据 self.x 的长度
    '''告诉 PyTorch 数据集包含多少样本，用于迭代或分割'''
'''TheDataset 是一个 PyTorch 数据集类, 支持有标签（训练/验证）和无标签（测试/预测）两种场景。特征和目标数据会被转换为浮点张量'''

'''
神经网络模型
'''

class My_Model(nn.Module):                      # 继承自 torch.nn.Module，定义神经网络的标准基类
    def __init__(self, input_dim):
        '''
        : input_dim: 输入特征的维度（整数），由外部传入，取决于数据集的特征数量'''
        super(My_Model, self).__init__()        # 调用父类 nn.Module 的构造函数，初始化模型
        # TODO: 修改模型结构, 注意尺寸
        self.layers = nn.Sequential(            # 使用 nn.Sequential 定义一个顺序层结构, 以下是包含的层
            nn.Linear(input_dim, 16),           # 线性层，将输入维度从 input_dim 映射到 16
            nn.ReLU(),                          # ReLU 激活函数，引入非线性
            nn.Linear(16, 8),                   # 线性层，将 16 维特征映射到 8 维
            nn.ReLU(),                          # 再次使用 ReLU 激活
            nn.Linear(8, 1)                     # 线性层，将 8 维特征映射到 1 维输出
        )

    def forward(self, x):                       # 前向传播
        '''
        : x: 输入张量，形状通常是 (B, input_dim)，其中 B 是批次大小
        '''
        x = self.layers(x)                      # 将输入 x 通过 self.layers 定义的层结构，得到输出张量，形状为 (B, 1)
        x = x.squeeze(1)                        # 移除张量的第 1 维（维度大小为 1) , 将 (B, 1) 变为 (B)
        return x
'''My_Model 是一个简单的全连接神经网络，输入维度为 input_dim，输出一个标量值（每个样本一个输出）'''
    
'''
选择特征
'''

def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    选择有用的特征进行回归
    : train_data：训练数据，通常是一个二维数组
    : valid_data：验证数据，格式与训练数据一致
    : test_data：测试数据，可能没有目标列
    : select_all：布尔值，默认为 True，控制是否选择所有特征
    '''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]               # 从训练数据和验证数据中提取最后一列作为目标变量 y_train 和 y_valid
    raw_x_train, raw_x_valid = train_data[:,:-1], valid_data[:,:-1]     # 从训练数据和验证数据中提取除最后一列外的所有列，作为特征 raw_x_train 和 raw_x_valid
    raw_x_test = test_data                                              # 测试数据直接赋值给 raw_x_test，假设其没有目标列（仅特征）

    if select_all:                                                      #
        feat_idx = list(range(raw_x_train.shape[1]))
    else:                                                               #
        feat_idx = [0,1,2,3,4] # TODO: 选择合适的特征列
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
     
'''
训练循环
'''

def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # 定义损失函数

    # 定义优化算法
    # TODO: 查看 https://pytorch.org/docs/stable/optim.html 可获得更多可用算法
    # TODO: L2 正则化(优化器(权重衰减......))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 

    writer = SummaryWriter() # 绘制tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # 创建保存模型的目录

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # 将模型设置为训练模式
        loss_record = []

        # tqdm 是一个可视化训练进度的进度条
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # 将梯度设为零
            x, y = x.to(device), y.to(device)   # 将数据移至设备
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # 计算梯度(反向传播)
            optimizer.step()                    # 更新参数
            step += 1
            loss_record.append(loss.detach().item())
            
            # 在 tqdm 进度条上显示当前的epoch数量和损耗。
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # 将模型设置为评估模式
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # 保存最佳模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

'''
配置
config包含用于训练的参数和保存模型的路径
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 优先使用GPU,  若没有, 使用CPU
config = {
    'seed': 1,      # 种子数字, 随意设置
    'select_all': True,   # 是否使用所有功能
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio验证规模 = 训练规模 * 有效比率
    'n_epochs': 3000,     # epochs数量           
    'batch_size': 256,    # batch大小
    'learning_rate': 1e-5,              
    'early_stop': 400,    # 如果模型在多轮epochs中没有改进, 则停止训练   
    'save_path': './models/model.ckpt'  # 保存模型的路径
}

'''
Dataloader
从文件中读取数据并设置训练集、验证集和测试集
'''

# 设置种子以实现可重复性
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# 打印数据大小
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# 输出特征的数量
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = TheDataset(x_train, y_train), \
                                            TheDataset(x_valid, y_valid), \
                                            TheDataset(x_test)

# Pytorch data loader 加载 pytorch dataset 到 batches 中
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
     
'''
开始训练
'''

model = My_Model(input_dim=x_train.shape[1]).to(device) # 将模型和数据放在同一设备上
trainer(train_loader, valid_loader, model, config, device)

'''
测试
将模型对测试集的预测结果存储在 pred.csv 中
'''

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred.csv')   