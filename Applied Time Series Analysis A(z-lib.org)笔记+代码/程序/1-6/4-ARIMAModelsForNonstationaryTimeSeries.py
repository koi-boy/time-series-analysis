'''
Example 1: ARIMA模型-以湖北省GDP为例
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA
# 1)数据绘图
time_series = pd.Series([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 396.26, 442.04, 517.77, 626.52, 717.08, 824.38, 913.38, 1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 3114.02, 3229.29, 3545.39, 3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61])
time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1978','2010'))
time_series.plot(figsize=(6,4))
plt.title("HuBei GDP(1978~2010)")
plt.show()

## 时间序列是呈指数形式的，波动性比较大，不是稳定的时间序列；对其取对数，将其转化为线性趋势。
# 2）转化
time_series = np.log(time_series)
time_series.plot(figsize=(6,4))
plt.title("HuBei GDP_log")
plt.show()

## 取对数之后的时间路径图明显具有线性趋势，为了确定其稳定性，对取对数后的数据进行ADF单位根检验
# 3)单位根检验
t=sm.tsa.stattools.adfuller(time_series, )
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

## 由上面的检验可知：t 统计量要大于任何置信度的临界值，因此认为该序列是非平稳的；故对序列进行差分处理，并进行 ADF 检验。
# 4）差分并进行ADF检验
time_series = time_series.diff(1)
time_series = time_series.dropna(how=any)
time_series.plot(figsize=(8,6))
plt.title("First-difference")
plt.show()
t=sm.tsa.stattools.adfuller(time_series)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)
## 差分之后的序列基本达到稳定，且通过了 ADF 检验。

## 确定自相关系数和平均移动系数（p,q）
## 根据时间序列的识别规则，采用 ACF 图、PAC 图，AIC 准则和 BIC 准则相结合的方式来确定 ARMA 模型的阶数, 应当选取 AIC 和 BIC 值达到最小的那一组为理想阶数。
# 5) p,q
plot_acf(time_series)
plot_pacf(time_series)
plt.show()

r,rac,Q = sm.tsa.acf(time_series, qstat=True)
prac = pacf(time_series,method='ywmle')
table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])

print(table)

#自动取阶p和q 的最大值，即函数里面的max_ar,和max_ma。
#ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的。

(p, q) =(sm.tsa.arma_order_select_ic(time_series,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
print((p,q))

# 6) ARIMA(0,1,1)
# p=0,q=2,一阶差分
p,d,q = (0,1,1)
arma_mod = ARMA(time_series,(p,d,q)).fit(disp=-1,method='mle')
summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))
print(summary)

# 7）白噪声检验
arma_mod = ARMA(time_series,(0,1,2)).fit(disp=-1,method='mle')
resid = arma_mod.resid
t=sm.tsa.stattools.adfuller(resid)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

plot_acf(resid)
plot_pacf(resid)
plt.show()

'''
Example 2: ARIMA模型-自行构建数据集
'''

import pandas as pd
import numpy as np
import seaborn as sns #热力图
import itertools
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf #画图定阶
from statsmodels.tsa.arima_model import ARIMA #模型
from statsmodels.tsa.arima_model import ARMA #模型
from statsmodels.stats.stattools import durbin_watson #DW检验
from statsmodels.graphics.api import qqplot #qq图

# 1)数据建立
def generate_data():
    index = pd.date_range(start='2018-1-1',end = '2018-9-1',freq='10T')
    index = list(index)
    data_list = []
    for i in range(len(index)):
        data_list.append(np.random.randn())
    dataframe = pd.DataFrame({'time':index,'values':data_list})
    dataframe.to_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/4-2-old_data.csv',index=0)
    print('the data is existting')
#generate_data()

# 2)数据处理
def data_handle():
    data = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/4-2-old_data.csv')
    print(data.describe())  # 查看统计信息,发现最小值有-10000的异常数据
    print((data.isnull()).sum())  # 查看是否存在缺失值
    print((data.duplicated()).sum())  # 重复值

    def change_zero(x):
        if x == -10000:
            return 0
        else:
            return x

    data['values'] = data['values'].apply(lambda x: change_zero(x))

    # 利用均值填充缺失值
    mean = data['values'].mean()

    def change_mean(x):
        if x == 0:
            return mean
        else:
            return x

    data['values'] = data['values'].apply(lambda x: change_mean(x))
    # 保存处理过的数据
    data.to_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/4-2-new_data0.csv', index=0)
    print('new data is existing')
#data_handle()

# 3）数据重采样
def Resampling():  # 重采样
    df = pd.read_csv('C:/Users/Dell/Desktop/Timeseries/project/Book_dataset/4-2-new_data0.csv')
    # 将默认索引方式转换成时间索引
    df['time'] = pd.to_datetime(df['time'])
    df.set_index("time", inplace=True)

    data = df['2018-1-1':'2018-8-1']  # 取18-1-1到8-1做预测
    test = df['2018-8-1':'2018-9-1']
    data_train = data.resample('D').mean()  # 以一天为时间间隔取均值,重采样
    data_test = test.resample('D').mean()

    return data_train, data_test


data_train, data_test = Resampling()

## 4）平稳性处理

# 4-1）差分法
def stationarity(timeseries):  # 平稳性处理
    # 差分法(不平稳处理),保存成新的列,1阶差分,dropna() 删除缺失值
    diff1 = timeseries.diff(1).dropna()
    diff2 = diff1.diff(1)  # 在一阶差分基础上做二阶差分

    diff1.plot(color='red', title='diff 1', figsize=(10, 4))
    # plt.show()
    diff2.plot(color='black', title='diff 2', figsize=(10, 4))
    # plt.show()

stationarity(data_train)

# 4-2）平滑法
timeseries = data_train
# 滚动平均（平滑法不平稳处理）
rolmean = timeseries.rolling(window=4, center=False).mean()
# 滚动标准差
rolstd = timeseries.rolling(window=4, center=False).std()

rolmean.plot(color='green', title='Rolling Mean', figsize=(10, 4))
plt.show()
rolstd.plot(color='blue', title='Rolling Std', figsize=(10, 4))
plt.show()

# 5)ADF检验
diff1 = timeseries.diff(1).dropna()

x = np.array(diff1['values'])
adftest = adfuller(x, autolag='AIC')
# print(adftest)

# 6)白噪声检验
p_value = acorr_ljungbox(timeseries, lags=1)
print (p_value)

# 7)定阶
def determinate_order(timeseries):
    #利用ACF和PACF判断模型阶数
    plot_acf(timeseries,lags=40) #延迟数
    plot_pacf(timeseries,lags=40)
    plt.show()

diff1 = timeseries.diff(1).dropna()
timeseries = diff1
determinate_order(timeseries)


### 构建模型和预测
(p, q) =(sm.tsa.arma_order_select_ic(diff1,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
print((p,q))

def ARMA_model(train, order):
    arma_model = ARMA(train, order)  # ARMA模型
    result = arma_model.fit()  # 激活模型
    print(result.summary())  # 给出一份模型报告
    ############ in-sample ############
    pred = result.predict()

    pred.plot()
    train.plot()
    print('标准差为{}'.format(mean_squared_error(train, pred)))

    # 残差
    resid = result.resid
    # 利用QQ图检验残差是否满足正态分布
    plt.figure(figsize=(12, 8))
    qqplot(resid, line='q', fit=True)
    plt.show()
    # 利用D-W检验,检验残差的自相关性
    print('D-W检验值为{}'.format(durbin_watson(resid.values)))
    return result

result = ARMA_model(diff1, (1, 1))
# 当D-W检验值接近于2时，说明模型较好。
