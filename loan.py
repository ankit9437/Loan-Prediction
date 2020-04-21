# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:51:52 2020

@author: DELL
"""

import numpy as np # for numpy functions and ndarray
import pandas as pd # for handling data
import matplotlib.pyplot as plt # for plotting graphs i.e. data visualisation
import seaborn as sns # for plotting graphs i.e. data visualisation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

Data=pd.read_csv("loan_train.csv")

# making a copy of the main Data to manipulate
X = Data.copy()
f = X.columns[6:-1]
# dataset description
X.describe(include="all")
#print(X)
X.Married.replace(['No', 'Yes'], [0, 1], inplace=True) # replacing 'No' with 0 and 'Yes' with 1 in 'Married' 
                                                       # column
    
X.Gender.replace(['M', 'F'], [0, 1], inplace = True) # replacing 'M' with 0 and 'F' with 1 in 'Gender' column

X.Dependents.replace(['0', '1', '2', '3+'], [0, 1, 2, 3], inplace=True) # replacing '0' with 0, '1' with 1, '2' 
                                                                        # with 2 and '3+' with 3 in
                                                                        # 'Dependents' column
        
X.Education.replace(['Not Graduate', 'Graduate'], [0, 1], inplace=True) # replacing 'Not Graduate' with 0 
                                                                        # and 'Graduate' with 1 in 'Education'
                                                                        # column
        
X.Self_Employed.replace(['No', 'Yes'], [0, 1], inplace=True) # replacing 'No' with 0 and 'Yes' with 1 in 
                                                             # 'Self_Employed' column

X.Property_Area.replace(['Rural','Semiurban', 'Urban'], [0, 1, 2], inplace=True) # replacing 'Rural' with 0, 
                                                                                 # 'Semiurban' with 1 and 
                                                                                 # 'Urban' with 2 in 
                                                                                 # 'Property_Area' column

X.Loan_Status.replace(['N', 'Y'], [0, 1], inplace = True) # replacing 'N' with 0 and 'Y' with 1 in 
                                                          # 'Loan_Status' column

#print(X.describe(include="all"))
#print(X.dtypes)
#print("Null values count:\n", X.isnull().sum()) 
X.Credit_History.fillna(0, inplace=True)

#filling Null values in binary columns with mode value
X.Self_Employed.fillna(X.Self_Employed.mode()[0], inplace=True)
X.Gender.fillna(X.Gender.mode()[0], inplace=True)
#filling null values with avg+std,avg-std
avg=X.LoanAmount.mean()
std=X.LoanAmount.std()
count=X.LoanAmount.isnull().sum()
ran=np.random.randint(avg-std,avg+std,size=count)
X['LoanAmount'][np.isnan(X['LoanAmount'])]=ran

avg=X.Loan_Amount_Term.mean()
std=X.Loan_Amount_Term.std()
cnt=X.Loan_Amount_Term.isnull().sum()
ran=np.random.randint(avg-std,avg+std,size=cnt)
X['Loan_Amount_Term'][np.isnan(X['Loan_Amount_Term'])]=ran

#sns.distplot(X["LoanAmount"])
xval=X.iloc[:,2:12].values
yval=X.iloc[:,12].values
#print(xval.shape)
#print(yval.shape)
final_Dataset = X.copy()
x_train,x_test,y_train,y_test=train_test_split(xval,yval,test_size=0.25,random_state=4)

#num_fold = 10
#seed = 7
#kfold = KFold(n_splits=num_fold, random_state=seed)


#result = cross_val_score(model, xval, yval, cv=kfold)
#print(result)
#print(result.mean()*100, result.std()*100)



classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
#print(cm)
print("Accuracy of f1_score by GaussianNB",f1_score(y_test,y_predict))
print("Accuracy Score of GaussianNB",accuracy_score(y_test,y_predict))
gnb=accuracy_score(y_test,y_predict)

classi=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classi.fit(x_train,y_train)
y_predict=classi.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
#print(cm)
print("Accuracy of f1_score by KNN",f1_score(y_test,y_predict))
print("Accuracy Score of KNN",accuracy_score(y_test,y_predict))
knn=accuracy_score(y_test,y_predict)

cl_entropy=DecisionTreeClassifier(criterion="entropy",random_state=10)
cl_entropy.fit(x_train,y_train)
y_pred=cl_entropy.predict(x_test)
#print(y_test,y_pred)
#for i in range(len(y_test)):
#    print(y_test[i],y_pred[i])
cm=confusion_matrix(y_test,y_pred)
#print(cm)
print("Accuracy of f1_score by DecisionTreeClassifier",f1_score(y_test,y_pred))
print("Accuracy Score of DecisionTreeClassifier",accuracy_score(y_test,y_predict))
dtree=accuracy_score(y_test,y_predict)

model=["GaussianNB","KNN","DecisionTree"]
accuracy=[gnb,knn,dtree]
barlist=plt.bar(model,accuracy,width=0.5,alpha=0.6)
plt.ylim(0,1.0)
barlist[0].set_color('r')
barlist[1].set_color('g')
barlist[2].set_color('y')
plt.xlabel('Classification Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy')
plt.grid()
plt.show()

data=pd.read_csv("loan_test.csv")
X=data.copy()
X.Property_Area.replace(['Rural','Semiurban','Urban'],[0,1,2],inplace=True)
xval=X.iloc[:,2:11].values
X.Credit_History.fillna(0,inplace=True)
X.Loan_Amount_Term.fillna(0,inplace=True)
avg=X.LoanAmount.mean()
std=X.LoanAmount.std()
count=X.LoanAmount.isnull().sum()
random=np.random.randint(avg-std,avg+std,size=count)
X['LoanAmount'][np.isnan(X['LoanAmount'])]=random

Naive_Bayes=GaussianNB()
Naive_Bayes.fit(final_Dataset[f],final_Dataset["Loan_Status"])
pred=Naive_Bayes.predict(X[f])
print(len(pred))
print(pred)


