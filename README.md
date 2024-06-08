House Price Prediction Using Linear Regression
This repository contains the implementation of a linear regression model to predict house prices based on their square footage, number of bedrooms, and number of bathrooms.

Table of Contents
Overview
Dataset
Installation
Usage
Model Training and Evaluation
Results
Contributing
License
Overview
The goal of this project is to predict house prices using a linear regression model. The features used for prediction are square footage, number of bedrooms, and number of bathrooms. The project involves data preprocessing, model training, and evaluation.

Dataset
The project uses two datasets:

train_dataset.csv: Contains the training data with features and target variable (price).
test_dataset.csv: Contains the test data with features. The target variable (price) is assumed to be available for evaluation purposes.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
Place your training and test datasets in the repository directory.
Update the file paths in the house_price_prediction.py script to point to your datasets.
Run the script:
bash
Copy code
python house_price_prediction.py
Model Training and Evaluation
The script house_price_prediction.py performs the following steps:

Loads the training and test datasets.
Fills missing values in numeric columns.
Defines the features (square_footage, bedrooms, bathrooms) and the target variable (price).
Trains a linear regression model using the training data.
Makes predictions on the test data.
Optionally evaluates the model using the actual prices for the test data (if available).
Script: house_price_prediction.py
python
Copy code
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
train_df = pd.read_csv('path_to_train_dataset.csv')
test_df = pd.read_csv('path_to_test_dataset.csv')

# Fill missing values only for numeric columns
numeric_cols_train = train_df.select_dtypes(include=[np.number]).columns
numeric_cols_test = test_df.select_dtypes(include=[np.number]).columns

train_df[numeric_cols_train] = train_df[numeric_cols_train].fillna(train_df[numeric_cols_train].mean())
test_df[numeric_cols_test] = test_df[numeric_cols_test].fillna(test_df[numeric_cols_test].mean())

# Define features and target variable with correct column names
X_train = train_df[['sqft', 'beds', 'baths']]
y_train = train_df['price']

X_test = test_df[['sqft', 'beds', 'baths']]

# Initialize the model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# If y_test (actual prices for test data) is available, calculate evaluation metrics
# For demonstration, let's assume you have y_test
# y_test = test_df['price']  # Uncomment if y_test is available

# Calculate evaluation metrics if y_test is available
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# Predict prices for new data (if any)
# new_data = pd.DataFrame({
#     'sqft': [new_square_footage_values],
#     'beds': [new_bedroom_values],
#     'baths': [new_bathroom_values]
# })
# predictions = model.predict(new_data)
# print(predictions)

# Output predictions for the test dataset
print(y_pred)
Results
The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.
The predicted prices for the test dataset are printed.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

License
This project is licensed under the MIT License.

Replace path_to_train_dataset.csv and path_to_test_dataset.csv with the actual file paths of your datasets, and update yourusername with your GitHub username.

You can save this content as a README.md file in your repository.








You’ve reached your GPT-4o limit.
