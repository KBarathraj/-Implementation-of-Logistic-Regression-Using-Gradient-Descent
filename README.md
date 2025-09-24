# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

## Program:
```python
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BARATHRAJ K
RegisterNumber:  212224230033

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt\

dataset=pd.read_csv('Placement_Data.csv')
dataset

dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')

dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y ):
    h = sigmoid(X.dot(theta)) 
    return -np.sum(y *np.log(h)+ (1- y) *np.log(1-h))
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot (h-y) /m
        theta-=alpha * gradient
    return theta
theta= gradient_descent (theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where( h >= 0.5,1 , 0)
    return y_pred

y_pred= predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y) 
print("Accuracy:", accuracy)
print(Y)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
print("Program executed Successfully")
```

## Output:
### Dataset:
<img width="1023" height="387" alt="0001" src="https://github.com/user-attachments/assets/1e964747-55d9-4ce7-bc0b-6a095885e723" />

### Labelling data:
<img width="393" height="381" alt="0002" src="https://github.com/user-attachments/assets/eb47eb7d-c20a-4ab7-a2b6-188fc885af95" />

### Labelling the column:
<img width="1049" height="479" alt="0003" src="https://github.com/user-attachments/assets/fa841d07-7acc-4959-a54e-e8c6c81092c0" />

### Dependent Variables:
<img width="1072" height="301" alt="0004" src="https://github.com/user-attachments/assets/c86ff66e-3ede-494c-93eb-57d09a8a805b" />

### Accuracy:
<img width="379" height="60" alt="0005" src="https://github.com/user-attachments/assets/3674e67d-ddec-4b27-86b7-240711764b32" />

### Y:
<img width="912" height="184" alt="0006" src="https://github.com/user-attachments/assets/226809f9-0c79-4497-8be6-16818d32bfd9" />

### New Predicted data:
<img width="695" height="343" alt="0007" src="https://github.com/user-attachments/assets/849a5bc4-cfae-4a6d-8bcd-7be43f3ff57b" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is developed and verified using python programming.

