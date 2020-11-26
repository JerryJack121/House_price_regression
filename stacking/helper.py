import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc
import numpy as np

train_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv')
val_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\valid-v3.csv')
test_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv')

# 合併所有資料，對年月日做onehot預處理
test_data['price'] = -1
data = pd.concat((train_data, val_data, test_data))
cate_feature = ['sale_yr', 'sale_month', 'sale_day']
a = ['sale_yr', 'sale_month']
for item in a:
    data[item] = LabelEncoder().fit_transform(data[item])
    item_dummies = pd.get_dummies(data[item])
    item_dummies.columns = [
        item + str(i + 1) for i in range(item_dummies.shape[1])
    ]
    data = pd.concat([data, item_dummies], axis=1)
data.drop(cate_feature, axis=1, inplace=True)

train = data[data['price'] != -1]
test = data[data['price'] == -1]

# # 清理內存
# del data, train_data, test_data
# gc.collect()

# 將不需要訓練的資料剃除
del_feature = ['id', 'price']
features = [i for i in train.columns if i not in del_feature]

# 分割feature與label
train_x = train[features]
train_y = train['price'].astype(int).values
test = test[features]

# 合併feature做正規化
scaler = StandardScaler()
all_features = pd.concat((train_x.iloc[:], test.iloc[:]))
all_features = scaler.fit_transform(all_features)
# 分割訓練、驗證、測試
val_x = all_features[train_data.shape[0]:train_data.shape[0] +
                     val_data.shape[0], ]
val_y = train_y[train_data.shape[0]:]
train_x = all_features[:train_data.shape[0], ]
train_y = train_y[:train_data.shape[0]]
test = all_features[train_data.shape[0] + val_data.shape[0]:, ]