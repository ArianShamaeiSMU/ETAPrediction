import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def Plot():
    # Plot the vertical bar chart
    plt.figure(figsize=(14, 8))
    plt.barh(list(sorted_etas.keys()), list(sorted_etas.values()))
    plt.title('Average Predicted ETAs for Each Category Level 1')
    plt.ylabel('Category Level 1')
    plt.xlabel('Average Predicted ETA (days)')
    plt.show()

# Load the dataset
file_path = '/Users/arianshamaei/code/ETAPrediction/Data/SMU IT support data proccessed.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Filter the data to only include Category Level 1 and Days to Resolve
data_filtered = data[['Category Level 1', 'Days to Resolve']]

# Convert categorical data to numerical data using one-hot encoding
data_filtered = pd.get_dummies(data_filtered, columns=['Category Level 1'])

# Separate features and target variable
X = data_filtered.drop('Days to Resolve', axis=1)
y = data_filtered['Days to Resolve']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, predictions)

# Predict ETAs for all categories in the dataset
all_categories_prediction = rf_model.predict(X)

# Add predictions back to the dataframe to associate with categories
data_filtered['Predicted ETA'] = np.concatenate([predictions, all_categories_prediction[len(predictions):]])

# Extract category names from the encoded features
category_names = X.columns.str.replace('Category Level 1_', '')

# Calculate average predicted ETA for each category
average_etas = {category: data_filtered[data_filtered[f'Category Level 1_{category}'] == 1]['Predicted ETA'].mean() for category in category_names}

# Sort average ETAs
sorted_etas = dict(sorted(average_etas.items(), key=lambda item: item[1]))

# Plot
Plot()

# Output the average ETAs for each category
print("Average Predicted ETAs for each Category Level 1:")
for category, eta in sorted_etas.items():
    print(f"{category}: {eta:.2f} days")