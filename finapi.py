import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load financial data into a pandas dataframe
data = pd.read_csv("financial_data.csv")

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data.drop(["Target"], axis=1), data["Target"], test_size=0.2)

# Initialize and train a linear regression model on the training data
model = LinearRegression()
model.fit(train_data, train_target)

# Use the trained model to make predictions on the test data
predictions = model.predict(test_data)

# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(test_target, predictions)
print("Mean Absolute Error:", mae)

# Create a function to generate personalized financial advice based on the user's input
def generate_advice(model, user_data):
    # Use the trained model to make a prediction on the user's inp
    prediction = model.predict(user_data)
    if prediction > 0:
        return "Based on your input, it appears that you are on track to reach your financial goals."
    else:
        return "Based on your input, it appears that you may need to make some changes to your financial plan to reach your goals."

# Example usage
user_data = {"income": 50000, "savings": 10000, "credit_score": 700}
print(generate_advice(model, user_data))
