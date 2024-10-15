# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: T.Roshini
RegisterNumber: 212223230175
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

![image](https://github.com/user-attachments/assets/9c995287-8006-48a9-886f-e93ec3367468)

![image](https://github.com/user-attachments/assets/edf8e6b2-b03a-4888-a2b1-c845b703f907)

![image](https://github.com/user-attachments/assets/1adb0dfd-3888-4dad-bc89-9c90554185ea)

![image](https://github.com/user-attachments/assets/79fb358b-0125-4ee7-8c58-d3753350379c)

![image](https://github.com/user-attachments/assets/2aa73530-2817-48cc-88b7-b50586657510)

![image](https://github.com/user-attachments/assets/48b76018-5f78-4f94-8ef0-cbe25039ffe9)

![image](https://github.com/user-attachments/assets/b363ab33-5dd5-40d2-b7f9-9c8cd2ed466b)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
