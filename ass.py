# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('accident_data.csv')  # Replace 'accident_data.csv' with your dataset file name

# Preprocessing
# Encode categorical variables
label_encoders = {}
for column in ['Road Conditions', 'Weather Conditions', 'Time of Day', 'Type of Vehicle Involved', 'Presence of Traffic Signals']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features and target variable
X = data.drop(columns=['Accident Severity'])
y = data['Accident Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')  # Save the model as 'accident_severity_model.pkl'
