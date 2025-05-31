import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Nonlineardat.csv")  # Make sure this file is in your working directory

# Extract Level as 2D array (column index 1)
X = df.iloc[:, 1:2].values

# Extract Salary as 1D array (last column)
y = df.iloc[:, -1].values

# Create polynomial features of degree 4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Train Linear Regression model on polynomial features
linreg = LinearRegression()
linreg.fit(X_poly, y)

# # Generate smooth range of levels for plotting the curve
# X_min = X.min()  # scalar minimum value
# X_max = X.max()  # scalar maximum value
# X_grid = np.arange(X_min, X_max + 0.1, 0.1).reshape(-1, 1)

# # Transform X_grid to polynomial features
# X_grid_poly = poly_reg.transform(X_grid)

# # Plot original data points
# plt.scatter(X, y, color='red', label='Actual Data')

# # Plot polynomial regression curve
# plt.plot(X_grid, lin_reg.predict(X_grid_poly), color='blue', label='Polynomial Regression')

# plt.title('Polynomial Regression: Position Level vs Salary')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Predict salary for a new level (6.5)
# level_to_predict = np.array([[6.5]])            # Input must be 2D array
# level_poly = poly_reg.transform(level_to_predict)  # Transform input to polynomial features
# predicted_salary = lin_reg.predict(level_poly)     # Predict

# # Print predicted salary (scalar value)
# print(f"Predicted salary for level 6.5: {predicted_salary[0]}")
 #FOR HIGHER RESOLUTION
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X,linreg.predict(X_poly),color='blue')
plt.title('POS VS SAL')
plt.xlabel('Pos level')
plt.ylabel('sal')
plt.show()