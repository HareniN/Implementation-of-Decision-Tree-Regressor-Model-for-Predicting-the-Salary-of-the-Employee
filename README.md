# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.


## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Hareni N

RegisterNumber:  212224040096
*/
```
import pandas as pd
data = pd.read_csv("/content/Salary.csv")


data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:

<img width="338" height="182" alt="image" src="https://github.com/user-attachments/assets/6dd67235-b434-4a71-9163-e294ae6142ea" />

--


<img width="147" height="168" alt="image" src="https://github.com/user-attachments/assets/540d238c-744d-44b9-851c-4271369bc5c7" />

--


<img width="267" height="209" alt="image" src="https://github.com/user-attachments/assets/6afd5542-9019-4d2d-8258-a4d7dd9d155f" />


--


<img width="135" height="31" alt="image" src="https://github.com/user-attachments/assets/003ac72b-7841-4a27-a057-14e2cdca7396" />


--


<img width="196" height="36" alt="image" src="https://github.com/user-attachments/assets/aa351751-f574-4569-abab-b411d11c402b" />


--


<img width="1475" height="58" alt="image" src="https://github.com/user-attachments/assets/409a1f40-77ce-49ac-acbe-1c00dc3d56c1" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
