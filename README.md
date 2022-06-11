# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```Python 
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shankar S S   
RegisterNumber:  21222120052 
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
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
Data Head:

![169464817-681d5776-e0a4-415a-bc58-10e5a9f37cc2](https://user-images.githubusercontent.com/93978702/173190470-c3b3adc8-4fe6-4b9f-a099-bc9dcd3aeaab.png)

Information:

![169464890-1a6c35c0-c7e5-43c8-9a40-2c6a0ee9a27b](https://user-images.githubusercontent.com/93978702/173190476-ec12d9bf-bc93-47ab-8b27-4892090f5722.png)

Null dataset:

![169464953-b46cc08c-2005-4acf-8fe7-08cae47aa60c](https://user-images.githubusercontent.com/93978702/173190484-3a343c0f-48b4-4251-b479-f70ec1e9aefd.png)

Accuracy:

![169465389-51117326-7f9b-416e-b48b-a43961d7f513](https://user-images.githubusercontent.com/93978702/173190492-3c2d1397-630f-4899-9f4e-3617e8a9b537.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
