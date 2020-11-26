from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import gc
from sklearn.model_selection import KFold
import torch.nn as nn
from helper import train_x, train_y, test
import sklearn.metrics as sm

dic = {}
y_pred_val_list = []
dic['id'] = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv').id
error=0
# KFold
kf = KFold(n_splits=5, shuffle=False)
fold = 0
for train_index, test_index in kf.split(train_x):
    trainfold_x = train_x[train_index]
    trainfold_y = train_y[train_index]
    valfold_x = train_x[test_index]
    valfold_y = train_y[test_index]
    fold += 1
    # create regressor object
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0, min_samples_leaf=1, max_features='auto')
    regressor.fit(trainfold_x, trainfold_y)
    y_pred_test = regressor.predict(test)
    y_pred_val = regressor.predict(valfold_x)

    error += sm.mean_absolute_error(valfold_y, y_pred_val)

    for i in range(len(y_pred_val)):
        y_pred_val_list.append(y_pred_val[i])

    dic['price' + str(fold)] = y_pred_test

print(error/5)
# 將測試集預測結果寫入 csv
price1_list = dic['price1']
price2_list = dic['price2']
price3_list = dic['price3']
price4_list = dic['price4']
price5_list = dic['price5']
avg_list = (price1_list + price2_list + price3_list + price4_list +
            price5_list) / fold

# 將5次測試結果csv
dic['price'] = avg_list
my_submission = pd.DataFrame(dic)
my_submission.to_csv('{}.csv'.format('./csv/forest_fold_test'), index=False)

# 將驗證集預測結果寫入csv
dic2 = {
    'id': pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').id,
    'realprice':
    pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').price,
    'price': y_pred_val_list
}
pd.DataFrame(dic2).to_csv('./csv/forest_fold_val.csv', index=False)