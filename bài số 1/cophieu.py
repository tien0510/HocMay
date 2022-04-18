from sklearn import linear_model
from matplotlib import pyplot
import numpy as np
import pandas as pd

data = pd.read_csv('Cophieu.csv')
# Dữ liệu huấn luyện
X= data.loc[0:74,['Open', 'PVolume', 'SVolume']].values
y= data.loc[0:74,['Close']].values
# Sử dụng  hồi quy tuyến tính
regr = linear_model.LinearRegression()
regr.fit(X, y)

#Dữ liệu test
Input = data.loc[75:,['Open', 'PVolume', 'SVolume']].values
def rep(Input):
    return str(Input).replace('[', '').replace(']', '')
Output = data.loc[75:,['Close']].values
def rep(Output):
    return str(Output).replace('[', '').replace(']', '')

w_0 = regr.intercept_   #sai số hồi quy
w_1 = regr.coef_ #hệ số hồi quy
print('Scikit-learn`s solution : hệ số hồi quy =',w_1,'sai số hồi quy = ',w_0)
for i in range(len(Input)):
    y1 = np.dot(w_1, Input[i]) + w_0
    def rep(y1):
        return str(y1).replace('[', '').replace(']', '')
    op = '{:,}'.format(int(rep(Input[i,[0]])))
    pv = '{:,}'.format(int(rep(Input[i,[1]])))
    sv = '{:,}'.format(int(rep(Input[i,[2]])))
    ge = float(rep(y1))
    GE = int(round(ge))
    ge = '{:,}'.format(int(round(ge)))
    cl = '{:,}'.format(int(rep(Output[i])))
    CL = int(rep(Output[i]))
    dl = CL -GE
    dl =  '{:,}'.format(dl)
    D = data.loc[75:, 'Date'].values
    GD.append(GE)
    GT.append(CL)
    print('Ngày',D[i], ',Giá Mở:', op,'VNĐ', ',KL mua:', pv, ',Kl bán:', sv, ',Giá Đoán:', ge,'VNĐ', ',Giá Thực tế:', cl, 'VNĐ', ',Độ lệch:', dl, 'VNĐ')
pyplot.xticks(rotation=90)
pyplot.plot(D,GD, color ='red',marker='.')
pyplot.plot(D,GT, color ='blue',marker='.')
pyplot.xlabel('Ngày')
pyplot.ylabel('Giá tiền(VNĐ)')
pyplot.legend(['Giá dự đoán','Giá thực tế'])
pyplot.show()
