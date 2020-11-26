from sklearn.svm import SVR
import pandas as pd
import gc
from sklearn.model_selection import KFold
from helper import train_x, train_y, test

dic = {}
y_pred_val_list = []
dic['id'] = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv').id

# KFold
kf = KFold(n_splits=5, shuffle=False)
fold = 0
for train_index, test_index in kf.split(train_x):
    trainfold_x = train_x[train_index]
    trainfold_y = train_y[train_index]
    valfold_x = train_x[test_index]
    fold += 1
    # 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
    rbf_svr = SVR(kernel='rbf', C=1e6)
    rbf_svr.fit(trainfold_x, trainfold_y)
    y_pred_test = rbf_svr.predict(test)
    y_pred_val = rbf_svr.predict(valfold_x)

    for i in range(len(y_pred_val)):
        y_pred_val_list.append(y_pred_val[i])

    dic['price' + str(fold)] = y_pred_test

# 將測試集預測結果寫入 csv
price1_list = dic['price1']
price2_list = dic['price2']
price3_list = dic['price3']
price4_list = dic['price4']
price5_list = dic['price5']
avg_list = (price1_list + price2_list + price3_list + price4_list +
            price5_list) / fold

# 將5次測試結果csv
dic['average'] = avg_list
my_submission = pd.DataFrame(dic)
my_submission.to_csv('{}.csv'.format('./csv/svm_fold_test_nodate'), index=False)

# 將驗證集預測結果寫入csv
dic2 = {
    'id': pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').id,
    'realprice':
    pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv').price,
    'price': y_pred_val_list
}
pd.DataFrame(dic2).to_csv('./csv/svm_fold_val_nodate.csv', index=False)

# ###################################################################################
# # # 驗證
print('R-squared value of RBF SVR is', rbf_svr.score(trainfold_x, trainfold_y))
# print 'The mean squared error of RBF SVR is', mean_squared_error(
#     ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
# print 'The mean absoluate error of RBF SVR is', mean_absolute_error(
#     ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
