import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce

df = pd.read_csv("job.csv")
data_X = df.iloc[:, :-1]
data_y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(data_X,data_y,test_size=0.3, shuffle = False)
encoder = ce.OrdinalEncoder(cols=['Gender','Stream'])
xtrain = encoder.fit_transform(xtrain)
xtest  = encoder.fit_transform(xtest)

svc = SVC(kernel = 'linear')

svc.fit(xtrain,ytrain)

data_x = svc.predict(xtest)
data_y = np.array(ytest)
count = 0
for i in range(len(data_x)):
    if(data_x[i] == data_y[i]):
        count = count + 1
    print(i,'Kết quả dự đoán :',data_x[i],', Thực tế :',data_y[i])
rate  = round((count/len(data_x))*100)
print('Support-Vector-Machine cho ta tỉ lệ dự đoán như sau : ')
print('Số dự đoán đúng',count ,'trên tổng',len(data_x),'\nTỷ lệ đúng đạt :' ,rate,'%' )
