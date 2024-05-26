# Simple Linear Regression Project

## Overview

This project demonstrates the use of a simple linear regression model to predict sales based on advertising budgets for TV. The goal is to understand the relationship between TV advertising expenditures and sales figures.

## Dataset

The dataset consists of 200 rows and 4 columns:
- `TV`: Advertising budget spent on TV
- `Radio`: Advertising budget spent on Radio
- `Newspaper`: Advertising budget spent on Newspaper
- `Sales`: Sales figures

## Steps

1. **Data Understanding and Preparation**
   - Load the dataset and explore its structure.
   - Identify the predictor variable (`TV`) and the target variable (`Sales`).

2. **Train-Test Split**
   - Split the data into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data.

3. **Model Training and Evaluation**
   - Train a simple linear regression model using the training data.
   - Scale the predictor and target variables using `StandardScaler` to standardize the data.
   - Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²).

4. **Interpretation and Visualization**
   - Interpret the model's coefficients to understand the relationship between TV advertising and sales.
   - Visualize the actual vs. predicted sales using scatter plots and the regression line.

## Code

### Libraries Used
- `pandas`: For data manipulation and analysis
- `matplotlib`: For data visualization
- `sklearn`: For machine learning tasks, including linear regression, scaling, and model evaluation

### Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = 'Simple linear regression (2).csv'
data = pd.read_csv(file_path)

# Define the features and the target
X = data[['TV']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Create LinearRegression object
lr_scaled = LinearRegression()

# Fit the model using the scaled training data
lr_scaled.fit(X_train_scaled, y_train_scaled)

# Check the parameters
intercept_scaled = lr_scaled.intercept_
coefficients_scaled = lr_scaled.coef_

# Print the parameters
print("Intercept (scaled):", intercept_scaled)
print("Coefficients (scaled):", coefficients_scaled)

# Make predictions
y_pred_scaled = lr_scaled.predict(X_test_scaled)

# Inverse transform the scaled predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the model on original scale
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.title('Sales vs TV Advertising')
plt.legend()
plt.show()
