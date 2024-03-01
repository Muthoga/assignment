# Load the saved model
loaded_model = joblib.load('accident_severity_model.pkl')

# Example of using the model to predict accident severity
example_data = {
    'Road Conditions': ['wet'],
    'Weather Conditions': ['rainy'],
    'Time of Day': ['morning'],
    'Type of Vehicle Involved': ['car'],
    'Speed Limit': [60],
    'Presence of Traffic Signals': ['yes']
}

# Preprocess the example data
for column, encoder in label_encoders.items():
    example_data[column] = encoder.transform(example_data[column])

# Convert example data to DataFrame
example_df = pd.DataFrame(example_data)

# Predict accident severity
predicted_severity = loaded_model.predict(example_df)
print(f"Predicted Accident Severity: {predicted_severity}")
