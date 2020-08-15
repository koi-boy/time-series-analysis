import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv("USB.csv")
data = data[::-1]
open = data['USB.Open']
high = data['USB.High']
low = data['USB.Low']
volume = data['USB.Volume']
adjusted = data['USB.Adjusted']
close = data['USB.Close']
plt.subplot(3,2,1)
plt.plot(open,close)
plt.subplot(3,2,2)
plt.plot(high,close)
plt.subplot(3,2,3)
plt.plot(low,close)
plt.subplot(3,2,4)
plt.plot(volume,close)
plt.subplot(3,2,5)
plt.plot(adjusted,close)
plt.show()

close = pd.read_csv("USB.csv",usecols=[4])
x = pd.read_csv("USB.csv",usecols=[1,2,3,5,6])
x = x[::-1]
close = close[::-1]
x_train = x[0:3401]
x_test = x[3401:]
y_train = close[0:3401]
y_test = close[3401:]
model = LinearRegression()
model.fit(x_train,y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b[0])

y_predict = []
x_test = np.array(x_test)
for i in range(5):
    y_predict.append((np.dot(b,x_test[i])+a)[0])

y_test = np.array(y_test)
error = mean_squared_error(y_test, y_predict)
for i in range(5):
    print("predicted:%f,expected:%f"%(y_predict[i],y_test[i]))
print('Test MSE: %.3f' % error)

predicted=[]
x = np.array(x)
for i in range(len(x)):
    predicted.append((np.dot(b,x[i])+a)[0])
plt.plot(np.array(close),label="expected")
plt.plot(predicted,label="predicted")
plt.legend()
plt.show()