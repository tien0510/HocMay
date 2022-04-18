import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import category_encoders as ce
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


ppn = Perceptron()
ppn.fit(xtrain, ytrain)
y_pred = ppn.predict(xtest)
y_test = np.array(ytest)

count = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
        print(i,'Dự đoán :', y_pred[i], ', Thực tế :', y_test[i])
rate  = round((count/len(y_pred))*100)
print('Perceptron cho ta tỉ lệ dự đoán như sau : ')
print('Số dự đoán đúng',count ,'trên tổng',len(y_pred),'\nTỷ lệ đúng đạt :' ,rate,'%' )







