import numpy as np # Numerical operations
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Data visualization
import pickle # Object serialization
from sklearn.datasets import fetch_california_housing # Dataset
from sklearn.model_selection import train_test_split # Data splitting
from sklearn.preprocessing import StandardScaler # Data scaling
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Evaluation metrics


housing = fetch_california_housing()


X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name=housing.target_names[0])


#X.to_csv('features.csv', index=False)
# Split data: 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
   X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)


# Calculate metrics for training set
train_perf = {'mse': mean_squared_error(y_train, y_train_pred),
             'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
             'mae': mean_absolute_error(y_train, y_train_pred),
             'r2': r2_score(y_train, y_train_pred)}


val_perf = {'mse': mean_squared_error(y_val, y_val_pred),
           'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
           'mae': mean_absolute_error(y_val, y_val_pred),
           'r2': r2_score(y_val, y_val_pred)}


### Answer (replace np.nan (not a number) with your actual MAPE calculations)
from sklearn.metrics import mean_absolute_percentage_error
train_perf['mape'] = mean_absolute_percentage_error(y_train, y_train_pred)
val_perf['mape'] = mean_absolute_percentage_error(y_val, y_val_pred)


for metric in train_perf.keys():
   print(f"{metric.upper():<5} - Train: {train_perf[metric]:.4f}, Validation: {val_perf[metric]:.4f}")


with open('checkpoint.pkl', 'wb') as f:
   pickle.dump({'model': model, 'scaler': scaler}, f)


