import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

 

df = pd.read_csv("job.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
encoder = ce.OrdinalEncoder(cols=['Gender','Stream'])
X = encoder.fit_transform(X)
# print(X)
# exit()
xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size = 0.8, shuffle = False)

#Xay dung mo hinh SVC
model_SVC = SVC()
model_SVC.fit(xtrain,ytrain)
output_SVC = model_SVC.predict(xtest)

#Xay dung mo hinh Gauss
model_GNB = GaussianNB()
model_GNB.fit(xtrain,ytrain)
output_GNB = model_GNB.predict(xtest)

#Xay dung mo hinh DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(xtrain,ytrain)
output_DTC = model_DTC.predict(xtest)

y_test = np.array(ytest)
d1 = d2 = d3 = 0

for i in range(len(output_SVC)):
        if output_SVC[i] == y_test[i]:
            d1 = d1 + 1
        # print(i,'Dự đoán :', output_SVC[i], ', Thực tế :', y_test[i])
rate_SVC  = round((d1/len(output_SVC))*100)
print('Support vector machine cho ta tỉ lệ dự đoán : ')
print('Số dự đoán đúng',d1 ,'trên tổng',len(output_SVC),'.Tỷ lệ đúng :' ,rate_SVC,'%\n')
for i in range(len(output_GNB)):
        if output_GNB[i] == y_test[i]:
            d2 = d2 + 1
        # print(i,'Dự đoán :', output_GNB[i], ', Thực tế :', y_test[i])
rate_GNB  = round((d2/len(output_GNB))*100)
print('Gaussian Naive Bayes cho ta tỉ lệ dự đoán : ')
print('Số dự đoán đúng',d2 ,'trên tổng',len(output_GNB),'.Tỷ lệ đúng :' ,rate_GNB,'%\n')
for i in range(len(output_DTC)):
        if output_DTC[i] == y_test[i]:
            d3 = d3 + 1
        # print(i,'Dự đoán :', output_DTC[i], ', Thực tế :', y_test[i])
rate_DTC  = round((d3/len(output_DTC))*100)
print('Decision Tree Classifier cho ta tỉ lệ dự đoán  : ')
print('Số dự đoán đúng',d3 ,'trên tổng',len(output_DTC),'.Tỷ lệ đúng :' ,rate_DTC,'%\n')
        


if( rate_SVC > rate_GNB and rate_SVC > rate_DTC):
    best_model = model_SVC
    best_score = rate_SVC

elif( rate_GNB > rate_SVC and rate_GNB > rate_DTC):
    best_model = model_GNB
    best_score = rate_DTC

elif( rate_DTC > rate_GNB and rate_DTC > rate_SVC):
    best_model = model_DTC
    best_score = rate_DTC
print("\nMô hình được chọn là : ",best_model)

import sys
from PyQt5.QtWidgets import QApplication,QMainWindow
from LNqtdesign import Ui_Dialog

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_Dialog()
        self.uic.setupUi(self.main_win)

        self.uic.btnresult.clicked.connect(self.showresult)
    def showresult(self):

        age = self.uic.sbage.value()
        gender = self.uic.cbgender.currentIndex()
        gender = gender + 1
        stream = self.uic.cbstream.currentIndex()
        stream = stream + 1
        intership = self.uic.sbintership.value()
        cpga = self.uic.sbcgpa.value()
        hostel = self.uic.cbhostel.currentIndex()
        hob = self.uic.cbhob.currentIndex()

        input = np.array([[age, gender, stream, intership, cpga, hostel, hob]])
# besst model
        output = best_model.predict(input)[0]
        if(output == 'YES') :
             output = 'Congratulations, you got the job'

        else :
            output = "Sorry, you didn't get the job"

        self.uic.Screen.setText(output)

    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())




