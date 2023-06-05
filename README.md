# EXPNO:05 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.


## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : CHANDRA SRINIVASULA REDDY 
RegisterNumber:  212220040028
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
1. Array value of x :

![image](https://user-images.githubusercontent.com/94175324/233020905-9823d741-33eb-49a7-9e77-dd4c418e8260.png)

2. Array value of y :

![image](https://user-images.githubusercontent.com/94175324/233021040-67f18e12-b0af-49fe-8c42-3510916ca5ad.png)

3. Exam 1 & 2 score graph :

![image](https://user-images.githubusercontent.com/94175324/233021299-a3b83c8e-7c29-4a13-aeb9-18f48fefd4e0.png)

4. Sigmoid graph :

![image](https://user-images.githubusercontent.com/94175324/233021506-eb8c1514-dbf0-4dc8-9717-a7fddacaf0d5.png)

5. J and grad value with array[0,0,0] :

![image](https://user-images.githubusercontent.com/94175324/233021938-9d0f74a2-21e8-440f-9afb-8e9de06d7368.png)

6. J and grad value with array[-24,0.2,0.2] :

![image](https://user-images.githubusercontent.com/94175324/233022154-daebb92c-a35b-4fc0-8c86-a6d1207b49aa.png)

7. res.function & res.x value :

![image](https://user-images.githubusercontent.com/94175324/233022342-5928b2d6-c825-47be-8462-a327763bb0c1.png)

8. Decision Boundary graph :

![image](https://user-images.githubusercontent.com/94175324/233022576-03651ec4-e2d8-4202-a15c-aa3dc9bc8561.png)

9. probability value :

![image](https://user-images.githubusercontent.com/94175324/233022876-e8c67fec-70f8-49b7-aa45-f5e966463182.png)

10. Mean prediction value :

![image](https://user-images.githubusercontent.com/94175324/233023043-622cba49-9f9b-43f3-9ffe-b11b12702791.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
