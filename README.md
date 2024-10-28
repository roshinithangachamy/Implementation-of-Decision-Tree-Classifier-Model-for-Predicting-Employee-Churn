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
![Screenshot 2024-10-15 201017](https://github.com/user-attachments/assets/1709f5c1-3715-4780-aeac-b0ca46d1c488)

#### Data Info
![Screenshot 2024-10-15 201102](https://github.com/user-attachments/assets/08ab380c-f68d-4f09-b650-1da2172f64e5)

#### Sum of Null Values
![Screenshot 2024-10-15 201153](https://github.com/user-attachments/assets/44548cee-d32b-4359-bfd6-3a34ed9728fd)

#### Labelling 
![Screenshot 2024-10-15 201218](https://github.com/user-attachments/assets/cb34ad1c-a293-4b6e-8573-169d1d862f16)

![Screenshot 2024-10-15 201629](https://github.com/user-attachments/assets/3cd47356-9621-4c2c-8f88-bbcd2e322e2f)

![Screenshot 2024-10-15 201705](https://github.com/user-attachments/assets/2c13b6b5-b17d-41f4-a381-5fcc83bb89c9)

![Screenshot 2024-10-15 201738](https://github.com/user-attachments/assets/e0ccb15a-6517-4b2e-98eb-79ddce56fdde)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
