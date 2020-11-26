import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocessing import preprocessing as pre

# 讀取預處理資料
train_val, train_x, val_x, test_x, train_y, val_y, ss_x, ss_y = pre()
# 將資料轉換為xgb格式
trn_data = xgb.DMatrix(train_x, label=train_y)
val_data = xgb.DMatrix(val_x, label=val_y)
watchlist = [(trn_data, 'train'), (val_data, 'valid')]
val_x = xgb.DMatrix(val_x)
test_x = xgb.DMatrix(test_x)
# 設定參數
num_round = 10000000
params = {
    'min_child_weight': 10.0,
    'learning_rate': 0.02,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round': 700,
    'nthread': -1,
    'missing': 1,
    'seed': 2019,
}
my_model = xgb.train(params,
                     trn_data,
                     num_round,
                     watchlist,
                     verbose_eval=20,
                     early_stopping_rounds=50)
# 驗證集之標記 label
val_y = ss_y.inverse_transform(val_y)
# 將驗證集丟入模型中進行預測
val_y_pred = my_model.predict(val_x)
val_y_pred = ss_y.inverse_transform(val_y_pred)
# 將測試集丟入模型中進行預測
y_pred = my_model.predict(test_x)
y_pred = ss_y.inverse_transform(y_pred)
# 輸出模型評估指標
print('r2_score', round(r2_score(val_y, val_y_pred), 4))
print('mean_squared_error', int(mean_squared_error(val_y, val_y_pred)))
print('mean_absolute_error', int(mean_absolute_error(val_y, val_y_pred)))

# 將結果寫入csv檔中
my_submission = pd.DataFrame({
    'id':
    pd.read_csv(r'../ntut-ml-regression-2020/test-v3.csv').id,
    'price':
    y_pred[:]
})
my_submission.to_csv('{}.csv'.format('../result/xgboost/blacky'), index=False)
