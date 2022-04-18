from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

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

#Implementing cross validation
k = int(input('Enter k  = '))
kf = KFold(n_splits=k, shuffle=False) # sử dụng K-Fold từ thư viện scikit-learn

acc_score = []
best_score = 0
i = 1

def rnd(i):
    input = round(i,2)
    return input
for train_index , test_index in kf.split(X):
    model = SVC(kernel='linear')
    print('test index :,',test_index)
    print('train index :',train_index)
    X_test, y_test = X.iloc[test_index, :], y[test_index]
    X_train , y_train = X.iloc[train_index,:],y[train_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)

    acc = accuracy_score(pred_values , y_test)*100
    print(i,'--',rnd(acc),'%')

    acc_score.append(acc)
    if  acc > best_score  :
        best_score = acc
        best_model = model
        turn          = i

        precision  = precision_score(pred_values, y_test, average='macro')
        recall     = recall_score(pred_values, y_test, average='macro')
        f_score    = f1_score(pred_values, y_test, average='macro')
    i = i +1


avg_acc_score = sum(acc_score)/k
print('Support-Vector-Machine với k-fold cross-validation :')
print('Avg accuracy  : ' ,rnd(avg_acc_score),'%')
print('Best accuracy : ' ,rnd(best_score),'%','lần thứ ',turn)
print('\nChất lượng mô hình dựa trên các độ đo :')
print("Precision     : " ,rnd(precision* 100),'%')
print("Recall        : " ,rnd(recall* 100),'%')
print("F1_score      : " ,rnd(f_score* 100),'%')

# input = np.array(X.iloc[100:200, :])
# guess = best_model.predict(input)
# for i in range(len(input)):
#     print(i,guess[i])

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




