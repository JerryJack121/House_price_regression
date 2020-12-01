## 作法說明
### 資料預處理
1. 刪除日，並將年、月資料做Onehot encoding 。
2. 合併所有資料做完正規化之後，再分割成訓練集、驗證集與測試集。
### 模型架構  
1. 先使用Adaboost、SVM、RandomForest及 XGBoost 四種弱回歸學習器分別對房價進行預測，並採用KFold交叉驗證5次，再將其預測結果作為輸入，以DNN模型融合4種輸出得到最終預測結果。
  
2. DNN架構如下所示：  
![NN架構1](https://github.com/JerryJack121/House_price_regression/blob/main/img/NN%E6%9E%B6%E6%A7%8B1.png?raw=true)
![NN架構2](https://github.com/JerryJack121/House_price_regression/blob/main/img/NN%E6%9E%B6%E6%A7%8B2.png?raw=true)
## 程式方塊圖與寫法
流程圖如下所示：  
  
![整題架構](https://github.com/JerryJack121/House_price_regression/blob/main/img/%E8%A8%93%E7%B7%B4%E6%9E%B6%E6%A7%8B.png?raw=true)  
  
DNN訓練分為兩階段，主要的差異在於優化器的不同，第一階段選用AdamW作為優化器，用於快速下降，第二階段選用SGD作為優化器，用於最後收斂。
## 結果分析
第一階段訓練結果如下：  
  
![AdamW訓練結果-1](https://github.com/JerryJack121/House_price_regression/blob/main/img/AdamW%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C-1.png?raw=true)
![AdamW訓練結果-2](https://github.com/JerryJack121/House_price_regression/blob/main/img/AdamW%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C-2.png?raw=true)
  
第二階段訓練結果如下：  
  
![SGD訓練結果](https://github.com/JerryJack121/House_price_regression/blob/main/img/SGD%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C.png?raw=true)
## 分析誤差原因
驗證損失無法繼續下降且已經收斂，推測可能是少了一些關鍵的特徵，或是特徵工程處理得不夠好。
## 如何改進？
進一步做特徵分析，特徵與特徵之間的關聯性比較。