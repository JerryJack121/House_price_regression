import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import torch
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv')
val_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\valid-v3.csv')

all_features = pd.concat((train_data, val_data))

price = all_features['price']
min_price = all_features['price'].min()
max_price = all_features['price'].max()
mean = all_features['price'].mean()
var = train_data['price'].std()
# print(min_price)

mean_list = []
var1_list = []
var2_list = []
var3_list = []
var_1_list = []
x = list(range(price.shape[0]))
for i in range(price.shape[0]):
    mean_list.append(mean)
    var1_list.append(mean + var)
    var2_list.append(mean + var * 2)
    var3_list.append(mean + var * 3)
    var_1_list.append(mean - var)

# plt.figure()
# plt.xlabel('data')
# plt.ylabel('price')
# plt.scatter(x, price)
# plt.plot(x, mean_list, 'r')
# plt.plot(x, var1_list, 'r--')
# plt.plot(x, var2_list, 'r--')
# plt.plot(x, var3_list, 'r--')
# plt.plot(x, var_1_list, 'r--')

# plt.legend(loc='best')
# plt.plot()
# plt.show()

bins = []

prices = []
distance = int((mean + var * 3) // 100)
new_price = 0
old_price = 0
while True:
    bins.append(new_price)
    new_price += distance
    if new_price > (mean + var * 3):
        break
    pri = int((new_price + old_price) / 2)
    prices.append(pri)
    old_price = new_price
bins.append(max_price)
prices.append(2500000)

# print(prices)
new_label = DataFrame()
new_label['Categories'] = pd.cut(price, bins, labels=prices)

labelencoder = LabelEncoder()
labelencoder = labelencoder.fit(prices)
print(labelencoder.classes_)
new_label['Categories'] = labelencoder.transform(new_label['Categories'])
# onehot_label = pd.get_dummies(new_label['Categories'])

newlabels = torch.tensor(new_label.Categories.values,
                         dtype=torch.float).view(-1, 1)

train_newlabels = newlabels[:12967]
val_newlabels = newlabels[12967:]

y = torch.LongTensor(new_label.Categories.values)

y = Variable(y)
