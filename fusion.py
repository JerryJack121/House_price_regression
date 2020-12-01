import pandas as pd
import numpy as np
import torch
from Net.model import fusion_net,JackNet,Net
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import copy
from tqdm import tqdm 

# 設定 GPU
device = torch.device("cuda")

# 載入數據
ada_df = pd.read_csv('./csv/adaboost_fold_val.csv')
svm_df = pd.read_csv('./csv/svm_fold_val.csv')
fore_df = pd.read_csv('./csv/forest_fold_val.csv')
knn_df = pd.read_csv('./csv/knn_fold_val.csv')
xgb_df = pd.read_csv('./csv/xgb_fold_val.csv')

train_x = torch.tensor(np.hstack((ada_df.values[:10000, 2:3], fore_df.values[:10000, 2:3], knn_df.values[:10000, 2:3], xgb_df.values[:10000, 2:3], svm_df.values[:10000, 2:3])),
                        dtype=torch.float).to(device)

val_x = torch.tensor(np.hstack((ada_df.values[10000:, 2:3], fore_df.values[10000:, 2:3], knn_df.values[10000:, 2:3], xgb_df.values[10000:, 2:3], svm_df.values[10000:, 2:3])),
                        dtype=torch.float).to(device)


train_y = torch.tensor(ada_df.values[:10000, 1:2], dtype=torch.float).to(device)
val_y = torch.tensor(ada_df.values[10000:, 1:2], dtype=torch.float).to(device)

ada_test_df = pd.read_csv('./csv/adaboost_fold_test.csv')
svm_test_df = pd.read_csv('./csv/svm_fold_test.csv')
fore_test_df = pd.read_csv('./csv/forest_fold_test.csv')
knn_test_df = pd.read_csv('./csv/knn_fold_test.csv')
xgb_test_df = pd.read_csv('./csv/xgb_fold_test.csv')
test = torch.tensor(np.hstack((ada_test_df.values[:, 6:7], fore_test_df.values[:, 6:7], knn_test_df.values[:, 6:7], xgb_test_df.values[:, 6:7], svm_test_df.values[:, 6:7])), 
                        dtype=torch.float).to(device)



# 訓練
model = fusion_net(features=train_x.shape[1])
# 使用 GPU 訓練
model.to(device)
# 定義損失函數
mae = nn.L1Loss(reduction='mean')
mse = nn.MSELoss(reduction='mean')
SMOOTHL1LOSS = torch.nn.SmoothL1Loss()
Loss = SMOOTHL1LOSS
# 選擇優化器
optimizer = torch.optim.AdamW(model.parameters(),
                                lr=1e-2)
         
scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.9)


# 設定訓練批次資料
batch_size = 5000
train_dataset = torch.utils.data.TensorDataset(train_x,
                                                train_y)
train_iter = torch.utils.data.DataLoader(train_dataset,
                                            batch_size,
                                            shuffle=True)

# 訓練與驗證
train_losses = []
val_losses = []
switch = True

# 計算初始損失
model.eval()  # 設定模型驗證模式
with torch.no_grad():
    optimizer.zero_grad()
    y_pred = model(val_x)
    loss = Loss(y_pred, val_y)
    print('初始loss:',loss.item())

num_epochs = 1000
for epoch in range(num_epochs):
    print('-' * 100)
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    # 訓練階段
    for train_betch_x, train_betch_y in train_iter:
        model.train()  # 設定模型訓練模式
        y_pred = model(train_betch_x)
        loss = Loss(y_pred, train_betch_y)
        if torch.isnan(loss):
            print('break')
            break
        train_running_loss = loss.item()
        # PyTorch的backward()方法計算梯度，預設會將每次的梯度相加，所以必須先歸零
        optimizer.zero_grad()
        # 反向傳播
        loss.backward()
        # 根據反向傳播得到的梯度更新模型參數
        optimizer.step() 
        scheduler.step()

        # 驗證階段
        model.eval()  # 設定模型驗證模式
        with torch.no_grad():
            optimizer.zero_grad()
            y_pred = model(val_x)
            loss = Loss(y_pred, val_y)
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
    train_losses.append(train_running_loss)
    val_losses.append(val_running_loss)
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print('train loss: {:4f} / val loss: {:4f} / Best_val_loss: {:4f} / best_epoch於 {} / Learning_rate: {}'.format(train_running_loss,
                                                                                            val_running_loss, 
                                                                                            best_val_loss,
                                                                                            best_epoch,
                                                                                            lr))
# 儲存模型
model.load_state_dict(best_model_wts)
torch.save(model ,'./weight/fusion_1127_adamw.pth')

# 繪出loss下降曲線
x = 100
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses[:x], label='train_losses')
plt.plot(val_losses[:x], label='val_losses')
plt.legend(loc='best')
plt.plot()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(range(x,num_epochs),train_losses[x:], label='train_losses')
plt.plot(range(x,num_epochs),val_losses[x:], label='val_losses')
plt.legend(loc='best')
plt.plot()
plt.show()

# 預測結果
model = torch.load('./weight/fusion_1127_adamw.pth')
with torch.no_grad():
    predictions = model(test).cpu().detach().numpy()
    my_submission = pd.DataFrame({
        'id':
        pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv').id,
        'price':
        predictions[:, 0]
    })
    my_submission.to_csv('./csv/fusion_1127_adamw.csv', index=False)
    print('寫入預測結果')