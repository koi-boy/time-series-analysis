import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_excel("data16_2.xlsx")
data2 = pd.read_excel("data16_2.xlsx",usecols=[5,6,7])
c = data.cons
i = data.inv
g = data.gov
x = data.other
y1 = []
y2 = []
y3 = []
t = []

for l in range(len(c)):
    t.append(l)
    y1.append(math.log(c[l]/x[l]))
    y2.append(math.log(i[l]/x[l]))
    y3.append(math.log(g[l]/x[l]))

plt.plot(y1,label="y1")
plt.plot(y2,label="y2")
plt.plot(y3,label="y3")
plt.legend()
plt.show()

dy1 = data.dy1
dy2 = data.dy2
dy3 = data.dy3
print(data)
orgMod = sm.tsa.VARMAX(data2,order=(3,0),exog=None)
#估计：就是模型
fitMod = orgMod.fit()
# 打印统计结果
print(fitMod.summary())
# 获得模型残差
resid = fitMod.resid
result = {'fitMod':fitMod,'resid':resid}