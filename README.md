# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```python
'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K KESAVA SAI
RegisterNumber:  212223230105
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="brown")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae= mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse= np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
### df.head() & df.tail()
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/83f52734-2202-471d-b3d2-8d3958a4992e)

### Array values of X
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/17526847-1529-4183-84f1-77d21115319b)

### Array values of Y
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/f5547e02-26da-430b-ab24-a5243c51767c)

### Predicted Values of Y
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/2b14f56d-6326-41a9-983c-58ab1b5cf5df)

### Test Values of Y
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/f3dc4f44-309d-493f-831e-a6e056bcb89c)

### Training Set Graph
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/a1c8f6d0-6b19-47b7-841c-7c0780abc24d)

### Testing Set Graph
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/24efdd8e-d4e2-4a51-bde1-1534dffb2e6d)

### Values of MSE, MAE & RMSE
![image](https://github.com/Kesavasai20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849303/60cfb119-6be0-4536-942b-d3d95ec3de99)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
