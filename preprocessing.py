import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


def one_hot(df, colNames):
    for col in colNames:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col])
        col_dummies = pd.get_dummies(df[col])
        col_dummies.columns = [
            col + str(i + 1) for i in range(col_dummies.shape[1])
        ]
        df = pd.concat([df, col_dummies], axis=1)

    df.drop(colNames, axis=1, inplace=True)

    return df


def PCA_function(all_features):
    pca = PCA(n_components=0.99)
    pca_features = pca.fit_transform(all_features)

    return pca_features


train_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\train-v3.csv')
val_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\valid-v3.csv')
test_data = pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv')

# 將所有資料合併做標準化
all_features = pd.concat(
    (train_data.iloc[:, 2:], val_data.iloc[:, 2:], test_data.iloc[:, 1:]))
all_features = one_hot(all_features, ['sale_yr', 'sale_month', 'sale_day'])
## normalized
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)
# all_features = all_features.apply(lambda x: (x - x.mean()) / (x.std()))
pca_features = all_features
# pca_features = PCA_function(all_features)

# 分割訓練資料與測試
n_train = train_data.shape[0]
n_val = val_data.shape[0]

train_features = torch.tensor(pca_features[:n_train], dtype=torch.float)
val_features = torch.tensor(pca_features[n_train:n_train + n_val],
                            dtype=torch.float)
test_features = torch.tensor(pca_features[n_train + n_val:], dtype=torch.float)

train_labels = torch.tensor(train_data.price.values,
                            dtype=torch.float).view(-1, 1)
val_labels = torch.tensor(val_data.price.values, dtype=torch.float).view(-1, 1)
