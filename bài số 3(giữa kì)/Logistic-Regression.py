import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('job.csv')
data_X = df.iloc[:, :-1]
data_y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(data_X,data_y,test_size=0.3, shuffle = False)

encoder = ce.OrdinalEncoder(cols=['Gender','Stream'])
xtrain = encoder.fit_transform(xtrain)
xtest  = encoder.fit_transform(xtest)
sc = StandardScaler()
sc.fit(xtrain)
xtrain = sc.transform(xtrain)
xtest = sc.transform(xtest)
# print(xtest)
# exit()
LogReg = LogisticRegression(solver = 'lbfgs')
LogReg.fit(xtrain,ytrain)
data_x = LogReg.predict(xtest)
m = np.array([[1.7,0.3,-0.4,1.6,0.6,1.8,-0.5,0.4]])

t = LogReg.predict(m)[0]
# print(t)
# exit()
data_y = np.array(ytest)
count = 0

for i in range(len(data_x)):
        if data_x[i] == data_y[i]:
            count = count + 1
        print(i,'Dự đoán :', data_x[i], ', Thực tế :', data_y[i])
rate  = round((count/len(data_x))*100)
print('Logistic-Regression cho ta tỉ lệ dự đoán như sau : ')
print('Số dự đoán đúng',count ,'trên tổng',len(data_x),'\nTỷ lệ đúng đạt :' ,rate,'%' )
