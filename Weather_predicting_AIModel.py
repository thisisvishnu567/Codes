# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic weather data for demonstration purposes
np.random.seed(42)
num_samples = 100
temperature = np.random.uniform(low=20, high=100, size=num_samples)
humidity = np.random.uniform(low=0, high=100, size=num_samples)
precipitation = np.random.uniform(low=0, high=10, size=num_samples)
#what is this??
target_variable = 30 * temperature + 20 * humidity + 10 * precipitation + np.random.normal(scale=10, size=num_samples)

# Create a DataFrame
weather_data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Precipitation': precipitation,
    'Target': target_variable
})

# Split the data into features (X) and target variable (y)
X = weather_data[['Temperature', 'Humidity', 'Precipitation']]
y = weather_data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Weather Prediction')
plt.show()
