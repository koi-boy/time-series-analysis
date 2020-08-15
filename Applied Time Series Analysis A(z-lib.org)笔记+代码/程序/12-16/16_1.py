import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("data16.xlsx")
x1 = data.x1
x2 = data.x2
x3 = data.x3
y1 = []
y2 = []
t = []

for i in range(len(x1)):
    y1.append(math.log(x1[i]/x3[i]))
    y2.append(math.log(x2[i]/x3[i]))
    t.append(i+1)

plt.plot(t,x1,label="BMI <25")
plt.plot(t,x2,label="BMI 25~30")
plt.plot(t,x3,label="BMI >30")

f1 = np.polyfit(t, y1, 3)
print('f1 is :\n',f1)

f2 = np.polyfit(t, y2, 3)
print('f2 is :\n',f2)

predict_t = [23,24,25,26,27,28,29]
predict_y1 = []
predict_y2 = []
for i in predict_t:
    predict_y1.append(np.polyval(f1,i))
    predict_y2.append(np.polyval(f2,i))
x1_x3 = []
x2_x3 = []
pre_x3 = []
pre_x1 = []
pre_x2 = []
for i in range(7):
    x1_x3.append(math.e**predict_y1[i])
    x2_x3.append(math.e**predict_y2[i])
    pre_x3.append(1/(1+x1_x3[i]+x2_x3[i])*100)
    pre_x1.append(pre_x3[i]*x1_x3[i])
    pre_x2.append(pre_x3[i]*x2_x3[i])

print(pre_x1)
plt.plot(predict_t,pre_x1,'--',label="BMI <25")
plt.plot(predict_t,pre_x2,'--',label="BMI 25~30")
plt.plot(predict_t,pre_x3,'--',label="BMI >30")
plt.legend()
plt.show()