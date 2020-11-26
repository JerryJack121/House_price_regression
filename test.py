from preprocessing import train_features, val_features, test_features
import pandas as pd
import numpy as np
import torch


# 房價預測結果寫入csv
def test_model(model, test_features):
    model = torch.load(model)

    with torch.no_grad():
        predictions = model(test_features).detach().numpy()
        my_submission = pd.DataFrame({
            'id':
            pd.read_csv(r'D:\DATASET\ntut-ml-regression-2020\test-v3.csv').id,
            'price':
            predictions[:, 0]
        })
        my_submission.to_csv('./csv/test04.csv', index=False)
    print('寫入預測結果')


def main():

    test_model('./weight/model_1117_mae81870.pth', test_features)


if __name__ == "__main__":
    main()
