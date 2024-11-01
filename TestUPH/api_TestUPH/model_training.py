# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Load the data
data = pd.read_csv('TMC_cleaning.csv')

# Define the features and target
X = data[['TESTER_ID', 'handler_id', 'product_no', 'QTY_IN', 'QTY_OUT']]
y = data['UPH']

# Identify categorical columns
categorical_features = ['TESTER_ID', 'handler_id', 'product_no']

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Leave numerical features as they are
)

# Create the pipeline with preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Create a 'model' folder to save the trained model
os.makedirs('app/model', exist_ok=True)

# Save the trained model using pickle
with open('app/model/rndf_regression_model.pkl', 'wb') as f:
	pickle.dump(pipeline, f)

print("Model trained and saved successfully.")