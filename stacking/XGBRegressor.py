import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
from helper import train_x, train_y, test
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold



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
    # 建立XGBRegressor模型
    my_model = XGBRegressor(n_estimators=2000, learning_rate=0.01)
    # 訓練模型
    my_model.fit(trainfold_x,
             trainfold_y,
             early_stopping_rounds=2000,
             eval_set=[(valfold_x, valfold_y)],
             verbose=True)

    # 將驗證集丟入模型中進行預測
    y_pred_val = my_model.predict(valfold_x)
    # 將測試集丟入模型中進行預測
    y_pred_test = my_model.predict(test)

    for i in range(len(y_pred_val)):
        y_pred_val_list.append(y_pred_val[i])

    dic['price' + str(fold)] = y_pred_test
    
# 輸出模型評估指標
print('r2_score', round(r2_score(valfold_y, y_pred_val), 4))
print('mean_squared_error', int(mean_squared_error(valfold_y, y_pred_val)))
print('mean_absolute_error', int(mean_absolute_error(valfold_y, y_pred_val)))

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
my_submission.to_csv('{}.csv'.format('./csv/xgb_fold_test'), index=False)

# 將驗證集預測結果寫入csv
dic2 = {
    'id': pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').id,
    'realprice':
    pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').price,
    'price': y_pred_val_list
}
pd.DataFrame(dic2).to_csv('./csv/xgb_fold_val.csv', index=False)
