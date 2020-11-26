import pandas as pd
import numpy as np
import torch
# from Net.model import fusion_net,JackNet,Net
import torch.nn as nn

# 設定 GPU
device = torch.device("cuda")

# 載入數據
ada_df = pd.read_csv('./csv/adaboost_fold_val.csv')
svm_df = pd.read_csv('./csv/svm_fold_val.csv')
fore_df = pd.read_csv('./csv/forest_fold_val.csv')
knn_df = pd.read_csv('./csv/knn_fold_val.csv')
xgb_df = pd.read_csv('./csv/xgb_fold_val.csv')

val_x = torch.tensor(np.hstack((ada_df.values[10000:, 2:3], fore_df.values[10000:, 2:3], knn_df.values[10000:, 2:3], xgb_df.values[10000:, 2:3], svm_df.values[10000:, 2:3])),
                        dtype=torch.float).to(device)
val_y = torch.tensor(ada_df.values[10000:, 1:2], dtype=torch.float).to(device)

mae = nn.L1Loss(reduction='mean')
model = torch.load('./weight/fusion_1125_2.pth')
with torch.no_grad():
    y_pred = model(val_x)
    loss = mae(y_pred, val_y)
    print('Loss:',loss.item())