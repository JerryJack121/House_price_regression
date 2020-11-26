import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Net.model import Net, Howard, JackNet
import copy
from torch.autograd import Variable
import time
from preprocessing import train_features, train_labels, val_features, val_labels
import tqdm
import sys
from torch.optim import lr_scheduler
# from analysis import train_newlabels, val_newlabels


def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.legend(loc='best')
    plt.plot()
    plt.show()


def train_model(train_features, train_labels, val_features, val_labels,
                num_epochs, batch_size):

    # 定義model
    # model = Net(features=train_features.shape[1])
    model = Howard(features=train_features.shape[1])
    # model = JackNet(features=train_features.shape[1])

    # 選擇損失函數
    mse = nn.MSELoss(reduction='mean')
    mae = nn.L1Loss(reduction='mean')
    CrossEntropy = nn.CrossEntropyLoss()
    NLLLoss = nn.NLLLoss()
    Loss = mae

    # 選擇優化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 設定訓練批次資料
    train_dataset = torch.utils.data.TensorDataset(train_features,
                                                   train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size,
                                             shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset,
                                           batch_size,
                                           shuffle=True)

    # 訓練
    train_losses = []
    val_losses = []
    switch = True

    for epoch in range(num_epochs):
        scheduler.step()
        t1 = time.time()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # 每個 batch 訓練階段
        for train_features, train_labels in train_iter:
            model.train()  # 設定模型訓練模式
            y_pred = model(train_features)
            loss = Loss(y_pred, train_labels)
            if torch.isnan(loss):
                break
            train_running_loss = loss.item()
            # PyTorch的backward()方法計算梯度，預設會將每次的梯度相加，所以必須先歸零
            optimizer.zero_grad()
            # 反向傳播
            loss.backward()
            # 根據反向傳播得到的梯度更新模型參數
            optimizer.step()
        # 驗證階段
        for val_features, val_labels in val_iter:
            model.eval()  # 設定模型驗證模式
            optimizer.zero_grad()
            y_pred = model(val_features)
            loss = Loss(y_pred, val_labels)
            val_running_loss = loss.item()
            # 複製最好的模型參數資料
            if switch == True:
                best_val_loss = val_running_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = 0
                switch = False
            if val_running_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print('最佳損失於第{:d}個epoch'.format(best_epoch))
        epoch_loss = train_running_loss
        train_losses.append(epoch_loss)
        epoch_val_loss = val_running_loss
        val_losses.append(epoch_val_loss)
        print('Loss: {:.4f} val_loss: {:.4f}'.format(epoch_loss,
                                                     epoch_val_loss))

        if torch.isnan(loss):
            break
        t2 = time.time()
        t = t2 - t1
        print('訓練時間: ', t)
    # 放入最好的模型參數輸出回傳
    print('\n最佳驗證損失', best_val_loss)
    model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses


def main():
    # 訓練模型
    model, train_losses, val_losses = train_model(train_features,
                                                  train_labels,
                                                  val_features,
                                                  val_labels,
                                                  num_epochs=200,
                                                  batch_size=64)

    # 儲存模型
    save_name = './weight/model_1117-1.pth'
    torch.save(model, save_name)
    print(save_name, '存檔')

    # 損失曲線
    plot_loss(train_losses, val_losses)


if __name__ == "__main__":
    main()
