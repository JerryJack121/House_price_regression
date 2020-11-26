import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
from Net.model import fusion_net as Net
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# 設定 GPU
device = torch.device("cuda")

# 讀取 csv 檔案
train_data = pd.read_csv('./dataset/train.csv', index_col=0)
val_data = pd.read_csv('./dataset/val.csv', index_col=0)
test_data = pd.read_csv('./dataset/test.csv', index_col=0)

train_x = train_data.iloc[:, 0:5].values
train_y = train_data.price.values
val_x = val_data.iloc[:, 0:5].values
val_y = val_data.price.values
test_x = test_data.iloc[:, 0:5].values

# 標準化
ss_x = StandardScaler()
ss_y = StandardScaler()

train_x = ss_x.fit_transform(train_x)
val_x = ss_x.transform(val_x)
test_x = ss_x.transform(test_x)

# train_y = train_y.reshape(-1, 1)
# val_y = val_y.reshape(-1, 1)

# train_y = ss_y.fit_transform(train_y)
# val_y = ss_y.transform(val_y)

# train_y = train_y.reshape(-1, )
# val_y = val_y.reshape(-1, )

# 轉換資料
train_x = torch.tensor(train_x, dtype=torch.float).to(device)
val_x = torch.tensor(val_x, dtype=torch.float).to(device)
test_x = torch.tensor(test_x, dtype=torch.float).to(device)
train_y = torch.tensor(train_y, dtype=torch.float).view(-1, 1).to(device)
val_y = torch.tensor(val_y, dtype=torch.float).view(-1, 1).to(device)

train_ls, val_ls = [], []

# pca降維
# pca = PCA(n_components=32)
# train_x = pca.fit_transform(train_x)
# val_x = pca.transform(val_x)
# test_x = pca.transform(test_x)

# 呼叫模型
model = Net(features=train_x.shape[1])
# 使用 GPU 訓練
model.to(device)
# 定義損失函數
# L1
criterion_L1 = nn.L1Loss(reduction='mean')
# SmoothL1
criterion_smoothL1 = nn.SmoothL1Loss(size_average=None,
                                     reduce=None,
                                     reduction='mean',
                                     beta=1.0)
# L2
criterion_L2 = nn.MSELoss(reduction='mean')

if True:
    # Hyperparameter（超參數）
    learning_rate = 8e-2
    num_epochs = 10000
    weight_decay = 0
    # 定義優化器
    optimizer = torch.optim.SGD(params=model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
    #                                            T_max=10000,
    #                                            eta_min=1e-6,
    #                                            last_epoch=-1)
    switch = True
    # 訓練模型
    train_ls, val_ls = [], []
    for epoch in range(num_epochs):
        # 切割資料
        train_x = train_x.clone().detach()
        val_x = val_x.clone().detach()
        test_x = test_x.clone().detach()
        train_y = train_y.clone().detach()
        val_y = val_y.clone().detach()

        # 切割訓練集
        batch_size = 4096
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_iter = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size,
                                                 shuffle=True)

        scheduler.step()
        with tqdm(train_iter) as pbar:
            for train_X, train_Y in train_iter:
                model.train()
                loss = criterion_L1(model(train_X.float()), train_Y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.eval()
                val_loss = criterion_L1(model(val_x.float()),
                                            val_y.float())
                # 複製最好的模型參數資料
                # if switch == True:
                #     best_val_loss = val_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())
                #     switch = False
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())
                pbar.update(1)
                pbar.set_description('{}/{}'.format(epoch + 1, num_epochs))
                pbar.set_postfix(
                    **{
                        '1_loss': loss.item(),
                        '2_val loss': val_loss.item(),
                        # '4_best_val_loss': best_val_loss.item(),
                        '5_lr': optimizer.state_dict()['param_groups'][0]['lr']
                    })

        train_ls.append(loss.item())
        val_ls.append(val_loss.item())

    # 讀取最好的權重
    model.load_state_dict(best_model_wts)
    # 儲存模型與權重
    torch.save(model, './logs/model.pth')

if False:
    # Hyperparameter（超參數）
    learning_rate = 1e-4
    num_epochs = 10000
    weight_decay = 0
    # 定義優化器
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
    #                                            T_max=10000,
    #                                            eta_min=1e-6,
    #                                            last_epoch=-1)
    model = model.float()
    switch = True
    # 訓練模型
    train_ls, val_ls = [], []
    for epoch in range(num_epochs):
        # 切割資料
        train_x = train_x.clone().detach()
        val_x = val_x.clone().detach()
        test_x = test_x.clone().detach()
        train_y = train_y.clone().detach()
        val_y = val_y.clone().detach()

        # 切割訓練集
        batch_size = 4096
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_iter = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size,
                                                 shuffle=True)
        scheduler.step()
        # scheduler.step()
        with tqdm(train_iter) as pbar:
            for train_X, train_Y in train_iter:
                loss = criterion_L2(model(train_X.float()), train_Y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    val_loss = criterion_L2(model(val_x.float()),
                                            val_y.float())
                    val_Y = val_y.detach().cpu().numpy()
                    val_Y = ss_y.inverse_transform(val_Y)
                    val_Y = torch.tensor(val_Y,
                                         dtype=torch.float).view(-1,
                                                                 1).to(device)

                    val_Y_pred = model(val_x.float()).detach().cpu().numpy()
                    val_Y_pred = ss_y.inverse_transform(val_Y_pred)
                    val_Y_pred = torch.tensor(
                        val_Y_pred, dtype=torch.float).view(-1, 1).to(device)
                    mae = criterion_L1(val_Y, val_Y_pred)
                # 複製最好的模型參數資料
                if switch == True:
                    best_mae = mae
                    best_model_wts = copy.deepcopy(model.state_dict())
                    switch = False
                if mae < best_mae:
                    best_mae = mae
                    best_model_wts = copy.deepcopy(model.state_dict())

                # if switch == True:
                #     best_val_loss = val_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())
                #     switch = False
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())

                pbar.update(1)
                pbar.set_description('{}/{}'.format(epoch + 1, num_epochs))
                pbar.set_postfix(
                    **{
                        '1_loss': loss.item(),
                        '2_val loss': val_loss.item(),
                        '3_mae': mae.item(),
                        '4_best_mae': best_mae.item(),
                        '5_lr': optimizer.state_dict()['param_groups'][0]['lr']
                    })

        train_ls.append(loss.item())
        val_ls.append(val_loss.item())

    # 讀取最好的權重
    model.load_state_dict(best_model_wts)
    # 儲存模型與權重
    torch.save(model, './logs/model.pth')

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_ls, label='train_losses')
plt.plot(val_ls, label='val_losses')
plt.legend(loc='best')
plt.savefig('./image/image.jpg')
plt.show()