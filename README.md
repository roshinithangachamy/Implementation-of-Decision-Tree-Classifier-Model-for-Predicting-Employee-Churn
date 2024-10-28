# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries such as pandas, sklearn, and numpy. Load the employee churn dataset and inspect it to understand the features and target (e.g., churn: yes/no)
2. Handle missing values, encode categorical variables using OneHotEncoder or LabelEncoder, and scale the data if necessary. Ensure the dataset is clean and prepared for training the model by handling outliers or inconsistent data.
3. Use train_test_split() from sklearn.model_selection to split the data. Divide it into training and testing sets (e.g., 80% training, 20% testing) for model evaluation.
4. Use DecisionTreeClassifier from sklearn.tree and fit the model on the training data. Set hyperparameters like max_depth or min_samples_split to avoid overfitting.
5. Use the trained decision tree model to predict employee churn on the test data. Store the predictions to compare them with the actual employee churn labels in the test set.
6. Calculate performance metrics such as accuracy, precision, recall, and F1-score using classification_report() from sklearn.metrics. These metrics will help evaluate how well the model predicts employee churn.
7. Use plot_tree() from sklearn.tree to visualize the decision tree structure. This provides insight into how the model makes decisions, helping interpret which features are most important for predicting churn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: T.Roshini
RegisterNumber: 212223230175
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
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

#### Dataset
![369336865-8e1afbe5-6cab-4f34-9f67-0c5363bef40d](https://github.com/user-attachments/assets/2605acd3-efed-4b95-b85b-779bad9636a5)

#### Data Info
![369336912-78022c77-d134-4d90-82ab-35bc151aa97b](https://github.com/user-attachments/assets/f95b5f97-c392-440a-aca7-31bd0f813db7)

#### Sum of Null
![369336979-c41453b2-93a5-4f5e-abbf-c8d8fe65d6a5](https://github.com/user-attachments/assets/89604b1c-7411-4fb4-859c-848df1786f16)
 Values

#### Labelling 
![369337020-52b8d38f-8b1b-441e-ab07-44b72c4c18a6](https://github.com/user-attachments/assets/1a4cc6b5-3a8e-4860-906f-4a845bd3ca12)

#### Assignment of x and y values
![369337263-655b6c0f-4a5d-40b9-a8e9-7cf70a6791bd](https://github.com/user-attachments/assets/6a5f2fc5-b959-463e-930c-3323da59beea)
![369337317-8272ff21-5841-4b4f-9df8-62e1f3dbea7a](https://github.com/user-attachments/assets/a7631a82-cd42-467b-9bf4-0767eec68cf9)

#### Converting string literals to numerical values using label encoder:
![369337407-032b7667-5476-4f90-9e63-937c0d33a8e9](https://github.com/user-attachments/assets/99bdc78b-ac78-42d4-a4de-caf07c30a4cb)

#### Accuracy
![369337452-44287cd3-cf56-4844-abe9-a5344f043ee7](https://github.com/user-attachments/assets/fc01cf8f-6180-4590-a88d-f271376bc17b)

#### Prediction
![369337486-1a7b1554-8bfb-40f2-bebf-78a6f0a010e4](https://github.com/user-attachments/assets/7d57f815-3744-4c4c-a172-d05268fb49ba)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
