## Part II Q

##Q1. [11 points] Use the entire dataset as the training set. Build a linear regression model with sklearn. Report the estimate of the coefficient for population.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
df = pd.read_csv('california_housing_random.csv')

# Define features and target
X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']

# Split dataset (though the task specifies to use the entire dataset, this step would still be useful in practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficient for population
population_coefficient = model.coef_[3]
print(f"Coefficient for population: {population_coefficient}")

#=================================================================================================================================================#
## Part II Q2
#Q2. [6 points] Use the model above to predict the following testing data point. 
# Create the testing data point
test_data = [[30, 2463, 444, 1000, 455, 4.7]]

# Make the prediction
predicted_value = model.predict(test_data)
print(f"Predicted median house value: {predicted_value[0]}")

#=================================================================================================================================================#

## Part II Q3
##Q3. [13 points] Use a kNN model with k=20 to make a prediction for the same data point above.
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the kNN model
knn_model = KNeighborsRegressor(n_neighbors=20)
knn_model.fit(X_scaled, y)

# Standardize the test data
test_data_scaled = scaler.transform(test_data)

# Make the prediction using kNN
knn_prediction = knn_model.predict(test_data_scaled)
print(f"kNN Predicted median house value: {knn_prediction[0]}")