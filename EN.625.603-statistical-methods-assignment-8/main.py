import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import sqrt

x = (30*7807.36 - 1300.69*323) / \
    sqrt(30*86754.6939 - 1300.69**2)/sqrt(30*11881 - 323**2)
print(x)

# # Your data
# # x = np.array([10, 4, 10, 15, 21, 29, 36, 51, 68]).reshape((-1, 1))
# x = np.array([10, 10.2, 10.2, 10.3, 10.3, 10.8, 11.0, 11.0, 11.2, 11.6, 12.1, 12.3,
#              12.6, 12.7, 12.9, 13, 13.9, 14.5, 14.7, 15.5, 16.4, 17.5, 18.1, 20.8, 22.4, 24.0]).reshape((-1, 1))
# y = np.array([88.7, 93.2, 95.1, 94, 88.3, 89.9, 67.7, 90.2, 95.5, 75.2, 84.6, 85,
#              94.8, 56.1, 54.5, 97.9, 83, 94, 91.4, 94.2, 97.2, 94.4, 78.6, 87.6, 93.3, 92.3])

# # Fit a linear regression model
# model = LinearRegression().fit(x, y)

# # Predict y values
# y_pred = model.predict(x)

# # Calculate residuals
# residuals = y - y_pred

# # Create a residual plot
# plt.scatter(x, residuals, color='blue')
# plt.hlines(0, min(x), max(x), colors='red', linestyles='dashed')
# plt.xlabel('Spending per Pupil')
# plt.ylabel('Graduation Rate')
# plt.show()
