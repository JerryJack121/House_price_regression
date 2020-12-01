import sklearn.tree as st
import sklearn.ensemble as se
import pandas as pd
import gc
from sklearn.model_selection import KFold
import numpy as np
import sklearn.metrics as sm
from helper import train_x, train_y, test


dic = {}
y_pred_val_list = []
dic['id'] = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv').id

# KFold
kf = KFold(n_splits=5, shuffle=False)
fold = 0
error=0
for train_index, test_index in kf.split(train_x):
    trainfold_x = train_x[train_index]
    trainfold_y = train_y[train_index]
    valfold_x = train_x[test_index]
    valfold_y = train_y[test_index]
    fold += 1
    # 创建基于决策树的正向激励回归器模型
    model = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=None,
                                                          splitter='best',
                                                          ),
                                 loss='linear',
                                 n_estimators=300,
                                 random_state=7,
                                 learning_rate= 1.3)
    # 训练模型

    model.fit(trainfold_x, trainfold_y)
    # # 测试模型
    y_pred_test = model.predict(test)
    y_pred_val = model.predict(valfold_x)
    for i in range(len(y_pred_val)):
        y_pred_val_list.append(y_pred_val[i])
    error += sm.mean_absolute_error(valfold_y, y_pred_val)
    dic['price' + str(fold)] = y_pred_test
print(error/5)
# # 將測試集預測結果寫入 csv
# price1_list = dic['price1']
# price2_list = dic['price2']
# price3_list = dic['price3']
# price4_list = dic['price4']
# price5_list = dic['price5']

# avg_list = (price1_list + price2_list + price3_list + price4_list +
#             price5_list) / fold

# # 將5次測試結果的平均寫入csv
# dic['price'] = avg_list
# my_submission = pd.DataFrame(dic)
# my_submission.to_csv('{}.csv'.format('./csv/adaboost_fold_test'), index=False)

# # 將驗證集預測結果寫入csv
# dic2 = {
#     'id': pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').id,
#     'realprice':
#     pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').price,
#     'price': y_pred_val_list
# }
# pd.DataFrame(dic2).to_csv('./csv/adaboost_fold_val.csv', index=False)