import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
x

y = np.array([40, 50, 50, 60, 70])
y

model = LinearRegression()
model.fit(x, y)

print("Intercept : ", model.intercept_)
print("Slope : ", model.coef_[0])

new_data = np.array([[6]])
predicted_mark = model.predict(new_data)

print("predicted mark : ", predicted_mark[0])
y_pred = model.predict(x)
y_pred

plt.scatter(x, y, color = "blue", label = "Actual Data")
plt.plot(x, y_pred, color = "red", label = "Regression Line")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours Vs. Marks")
plt.grid(True)
plt.legend()
plt.show()


from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("Mean Absolute Error : ", mae)
print("Mean Squared Error : ", mse)
print("Root Mean Squared Error : ", rmse)
print("R-squared : ", r2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42)

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas = [0.01, 0.1, 1, 10, 100])
)

model.fit(x_train, y_train)

ridge_model = model.named_steps["ridgecv"]
print("Best alpha : ", ridge_model.alpha_)

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha = 0.1)
    )

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

ridge_step = model.named_steps['ridge']

print("Intercept : ", ridge_step.intercept_)
print("Standard coefficients : ", ridge_step.coef_)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("MAE       : ", mean_absolute_error(y_test, y_pred))
print("MSE       : ", mean_squared_error(y_test, y_pred))
print("RMSE      : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared : ", r2_score(y_test, y_pred))