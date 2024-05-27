import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
products_df = pd.read_json('Products.json')
category_df = pd.read_csv('category.csv')
store_df = pd.read_csv('store.csv')
subcategory_df = pd.read_csv('subcategory.csv')
sales_df = pd.read_csv('sales.csv', parse_dates=['Date'], dayfirst=True)

# how many uniue counts of Store_ID in sales_df 
sales_df['Store_ID'].nunique()
#print(sales_df)


# Check the first few rows of each data frame
# print(products_df.head())
# print(category_df.head())
# print(store_df.head())
# print(subcategory_df.head())
# print(sales_df.head())

# # Summary statistics
# print(sales_df.describe())

# # Check for missing values
# print(sales_df.isnull().sum())

# type of all columns of sales_df
#print(sales_df.dtypes)

# Convert the 'Date' column to DateTime
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

print(sales_df.dtypes)
#Rename ' Discount_percent ' column to 'Discount_percent'
sales_df.rename(columns={' Discount_percent ': 'Discount_percent'}, inplace=True)

# Fill missing values or drop rows/columns if necessary
sales_df['Promotion_Flag'].fillna(0, inplace=True)  # Assuming no promotion if null
sales_df['Discount_percent'].fillna(0, inplace=True)  # Assuming no discount if null

print(sales_df.head())
sales_df.dropna(inplace=True)  # Drop rows with missing values

# Convert data types if needed
sales_df['Discount_percent'] = sales_df['Discount_percent'].astype(float)

#check total number of rows in sales_df
print(sales_df.count())
sales_df_first = sales_df
#sales_df = sales_df_first

# Aggregate sales data by date
daily_sales = sales_df.groupby('Date')['Sales_Volume'].sum().reset_index()

# Set the date as the index
daily_sales.set_index('Date', inplace=True)

# Resample to monthly sales volume
monthly_sales = daily_sales.resample('M').sum()

#print(monthly_sales.head())

# XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def lagged_data(data, lag=1):
    data_lagged = pd.concat([data.shift(i) for i in range(lag, -1, -1)], axis=1)
    data_lagged.columns = [f'X-{i}' for i in range(lag, -1, -1)]
    data_lagged = data_lagged.dropna()
    return data_lagged

# Create lagged data
lag = 3
data_lagged = lagged_data(monthly_sales, lag=lag)
X = data_lagged.drop('X-0', axis=1)
y = data_lagged['X-0']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

# Fit the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
#calculate r2_score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f'R2: {r2}')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(y, label='Actual')
plt.plot(pd.Series(y_pred, index=y_test.index), label='Predicted')
plt.legend()
plt.show()

# Aggregate sales data by date
daily_sales = sales_df.groupby('Date')['Sales_Volume'].sum().reset_index()

# Set the date as the index
daily_sales.set_index('Date', inplace=True)

# Resample to monthly sales volume
monthly_sales = daily_sales.resample('M').sum()

# Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit the model
model = ExponentialSmoothing(monthly_sales, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Make predictions
y_pred = model_fit.predict(start=0, end=len(monthly_sales) + 12)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(monthly_sales, y_pred[:len(monthly_sales)]))
print(f'RMSE: {rmse}')
#calculate r2_score
from sklearn.metrics import r2_score
r2 = r2_score(monthly_sales, y_pred[:len(monthly_sales)])
print(f'R2: {r2}')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# Aggregate sales data by date
daily_sales = sales_df.groupby('Date')['Sales_Volume'].sum().reset_index()

# Set the date as the index
daily_sales.set_index('Date', inplace=True)

# Resample to monthly sales volume
monthly_sales = daily_sales.resample('M').sum()

# logistc regression model for sales prediction
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Create lag features for the Linear Regression model
def create_lag_features(data, lags):
    lagged_data = pd.DataFrame(index=data.index)
    lagged_data['target'] = data.values
    for lag in lags:
        lagged_data[f'lag_{lag}'] = data.shift(lag)
    return lagged_data.dropna()

# Define lags
lags = [1, 3, 6, 12]

# Create lagged features
lagged_data = create_lag_features(monthly_sales, lags)

# Split the data into train and test sets
train_data, test_data = train_test_split(lagged_data, test_size=0.2, shuffle=False)

# Separate features and target variable
X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Combine train and test predictions
y_pred = np.concatenate([y_pred_train, y_pred_test])

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(lagged_data['target'], y_pred))
print(f'RMSE: {rmse}')

# Calculate the R2 score
r2 = r2_score(lagged_data['target'], y_pred)
print(f'R2: {r2}')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, label='Actual')
plt.plot(lagged_data.index, y_pred, label='Predicted')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Aggregate sales data by date
daily_sales = sales_df.groupby('Date')['Sales_Volume'].sum().reset_index()

# Set the date as the index
daily_sales.set_index('Date', inplace=True)

# Resample to monthly sales volume
monthly_sales = daily_sales.resample('M').sum()

# Create lag features for the Ridge Regression model
def create_lag_features(data, lags):
    lagged_data = pd.DataFrame(index=data.index)
    lagged_data['target'] = data.values
    for lag in lags:
        lagged_data[f'lag_{lag}'] = data.shift(lag)
    return lagged_data.dropna()

# Define lags
lags = [1, 2, 3, 4, 5, 6, 12]

# Create lagged features
lagged_data = create_lag_features(monthly_sales, lags)

# Split the data into train and test sets
train_data, test_data = train_test_split(lagged_data, test_size=0.2, shuffle=False)

# Separate features and target variable
X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# Fit the Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Combine train and test predictions
y_pred = np.concatenate([y_pred_train, y_pred_test])

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(lagged_data['target'], y_pred))
print(f'RMSE: {rmse}')

# Calculate the R2 score
r2 = r2_score(lagged_data['target'], y_pred)
print(f'R2: {r2}')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, label='Actual')
plt.plot(lagged_data.index, y_pred, label='Predicted')
plt.legend()
plt.show()

