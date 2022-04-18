from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import category_encoders as ce
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
 

df = pd.read_csv("job.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
encoder = ce.OrdinalEncoder(cols=['Gender','Stream'])
X = encoder.fit_transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size = 0.8, shuffle = False)

#Xay dung mo hinh
model = SVC()
model.fit(xtrain,ytrain)

data = model.predict(xtest)
y_test = np.array(ytest)
count = 0

for i in range(len(data)):
        if data[i] == y_test[i]:
            count = count + 1
        print(i,'Dự đoán :', data[i], ', Thực tế :', y_test[i])
        
rate  = round((count/len(data))*100)
print('\nsupport vector machine cho ta tỉ lệ dự đoán như sau : ')
print('Số dự đoán đúng',count ,'trên tổng',len(data),'\nTỷ lệ đúng đạt :' ,rate,'%')
print(model)
    

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




