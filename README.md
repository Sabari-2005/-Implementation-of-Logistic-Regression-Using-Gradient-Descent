# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SABARINATH R
RegisterNumber:  212223100048
*/
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("placement.csv")  
data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 
data
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data 
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values
y 
theta = np.random.randn(x.shape[1]) 
Y=y 
def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,X,y,alpha,num_iterations): 
    m=len(y)
    for i in range(num_iterations): 
        h=sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h-y)/m 
        theta-=alpha * gradient 
        return theta
gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000) 
def predict(theta,X): 
    h=sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred 
y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy) 
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew)

```

## Output:
## Dataset :
![image](https://github.com/user-attachments/assets/aba6fdad-56c7-4c65-93ba-ac98af1c69fa)


## Information :
![image](https://github.com/user-attachments/assets/287f6b85-e9fc-4bc4-a65a-89af6aa7934b)

## Encoding:
![image](https://github.com/user-attachments/assets/f3ce83d8-4c15-4001-b4ce-652a0137e840)

## X and Y value:
![image](https://github.com/user-attachments/assets/14f65931-07b3-4d74-88c5-8212ab7f8736)
![image](https://github.com/user-attachments/assets/6d970b42-8938-4ac9-975f-6758297eef68)

## Gradient Descent:
![image](https://github.com/user-attachments/assets/f4fa959d-942a-4956-ab8e-e3b186ca371a)

## Accuracy:
![image](https://github.com/user-attachments/assets/cb0730f1-0efc-4e3e-af52-c2c0ecbbb35d)

## Prediction:
![image](https://github.com/user-attachments/assets/f819055d-7d40-4308-b2eb-1ce13b2c3644)
![image](https://github.com/user-attachments/assets/649125d0-928c-4b29-abf2-4f215c3bebf2)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

