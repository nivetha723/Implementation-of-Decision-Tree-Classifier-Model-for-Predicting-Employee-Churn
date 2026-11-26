# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the employee churn dataset.
3. Split the dataset into training and testing data.
4. Train the Decision Tree Classifier model and predict employee churn.
5. Evaluate the model using accuracy score and display the decision tree.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Nivetha N 
RegisterNumber:  212225040290
*/
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("Employee.csv")

print("First five records of the dataset:")
print(data.head())

X = data[
    [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years"
    ]
]

y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nActual Churn Status:")
print(y_test.values[:20])

print("\nPredicted Churn Status:")
print(y_pred[:20])
print("\nAccuracy:", accuracy)

print("\nAccuracy Percentage:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_employee = pd.DataFrame({
    "satisfaction_level": [0.35],
    "last_evaluation": [0.80],
    "number_project": [5],
    "average_montly_hours": [250],
    "time_spend_company": [5],
    "Work_accident": [0],
    "promotion_last_5years": [0]
})


prediction = model.predict(new_employee)

print("\nNew Employee Details:")
print("Satisfaction Level: 0.35")
print("Last Evaluation: 0.80")
print("Number of Projects: 5")
print("Average Monthly Hours: 250")
print("Time Spent in Company: 5 years")
print("Work Accident: 0")
print("Promotion in Last 5 Years: 0")

if prediction[0] == 1:
    print("\nPredicted Churn Status: Employee will leave")
else:
    print("\nPredicted Churn Status: Employee will stay")
```

## Output:
<img width="472" height="631" alt="image" src="https://github.com/user-attachments/assets/c2895474-b712-48a2-887d-73ba3eaed000" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
