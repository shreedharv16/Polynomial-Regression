# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#building a simple linear reg model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#building a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y) 

#visualizing the linear regression results
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression Model')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression Model')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Predicting the result using lin reg
lin_reg.predict([[9]]) #years of experience

#Predicting the result using poly reg
lin_reg2.predict(poly_reg.fit_transform([[9]]))
