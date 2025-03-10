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
    '''修正随机数生成器种子，以提高可重复性'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''将提供的训练数据分成训练集和验证集'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # 将模型设置为评估模式
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

'''
Dataset
'''

class TheDataset(Dataset):
    '''
    x: 特征
    y: 目标，如果没有，则进行预测
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
'''
神经网络模型
'''

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改模型结构，注意尺寸
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
    
'''
选择特征
'''

def select_feat(train_data, valid_data, test_data, select_all=True):
    '''选择有用的特征进行回归'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
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
config包含用于训练的超参数和保存模型的路径
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 1,      # 种子数, 随意设置
    'select_all': True,   # 是否使用所有功能
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio验证规模 = 训练规模 * 有效比率
    'n_epochs': 3000,     # epochs数量           
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # 如果模型在多轮epochs中没有改进，则停止训练   
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