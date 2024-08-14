sepal-length 꽃받침 길이
sepal-width 꽃받침 너비
petal-length 꽃잎 길이
patal-width 꽃잎 너비
class: 꽃의 종류
# (1)주어진 데이터 셋에 대한 정보 파악(데이터요약(describe()) 및 시각화)
# (2)꽃받침 및 꽃잎의 정보를 바탕으로 꽃의 종류를 예측하는 의사결정나무 모델 개발
# (3)개발한 모델의 성능 평가:KFold 교차검증 방법을 활용(정확도)
# (4)개발한 모델의 성능 평가: 오차행렬, 정밀도, 재현율 등

# 모델 예측
y_pred = model_predict(X_test)
confusion_matrix(y_pred, Y_test)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/1.salary.csv')
array = data.values
X = array[:,1]
Y = array[:,1]
fig, ax = plt.subplots()
plt.clf()
plt.scatter(X,Y, label='random',color='gold',marker='*',
            s=30, alpha=0.5)
X1=X.reshape(-1,1)


(X_train,X_text,
 Y_train,Y_test)=train_test_split(X1,Y,test_size=0.2)
model =LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_text)
print(y_pred)

plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)),Y_test,color='blue',
            marker='o')
plt.plot(range(len(y_pred)),y_pred,color='r'
            ,marker='x')
plt.show()

mae=mean_absolute_error(y_pred,Y_test)
print(mae)