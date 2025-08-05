############################################################################
# BUS ADM 812-001: machine Learning For Business                           #   
# HW5: Predicting Airbnb Prices Using Linear Regression and kNN            #
# BY~ Sandeep Seelam, Abhishek Yadav, Sai Krishna Chaitanya, Sohini Sanyal #
############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Task 1: Function to find the optimal k using 5-fold cross-validation
def find_optimal_k(X_std, y, visualize=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_range = range(2, 101)  # Values of k to test
    sse = []  # To store SSE for each k
    
    for k in k_range:
        sse_k = 0
        # Perform 5-fold cross-validation
        for train_index, test_index in kf.split(X_std):
            X_train, X_test = X_std[train_index], X_std[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Initialize and train kNN regressor
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            # Compute SSE for this fold
            sse_k += mean_squared_error(y_test, y_pred) * len(y_test)
        
        # Average SSE for this k
        sse.append(sse_k / len(y))
    
    # Find the k with the minimum SSE
    k_best = k_range[np.argmin(sse)]
    
    # Plot x=k vs y=SSE if visualize is specified
    if visualize:
        plt.plot(k_range, sse)
        plt.xlabel('k')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('SSE vs k')
        plt.savefig(visualize)
        plt.show()
    
    return k_best

# Task 1: Function to predict prices using sklearn's KNN
def knn_sklearn(X, y, X_pred):
    # Standardize the predictors
    scaler = MinMaxScaler()
    X_std = scaler.fit_transform(X)
    X_pred_std = scaler.transform(X_pred)

    # Find the optimal k
    k_best = find_optimal_k(X_std, y, visualize='hw6.png')  # Save plot as hw6.png

    # Use the optimal k to create the final kNN regressor
    knn = KNeighborsRegressor(n_neighbors=k_best)
    knn.fit(X_std, y)

    # Predict the values for X_pred
    y_pred = knn.predict(X_pred_std)

    return y_pred, k_best

# Task 2: Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    distance = 0.0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return np.sqrt(distance)

# Task 2: Manually implemented KNN algorithm
def knn(k, X, y, X_pred):
    preds = []
    for pred_point in X_pred:
        # Calculate distances between pred_point and all points in X
        distances = np.array([euclidean_distance(pred_point, x_point) for x_point in X])
        
        # Get the indices of the k nearest neighbors
        k_nearest_indices = distances.argsort()[:k]
        
        # Get the average of the y values of these neighbors
        pred_value = np.mean(y[k_nearest_indices])
        preds.append(pred_value)
    
    return np.array(preds)

# Load the dataset
df = pd.read_csv('susedcars.csv', usecols=['price', 'mileage', 'year'])
df['age'] = 2015 - df.pop('year')  # Calculate age of the cars
X = df[['mileage', 'age']].to_numpy()  # Predictor variables
y = df['price'].to_numpy()  # Target variable (price)

# Predictions for cars with mileage=100000, age=10 and mileage=50000, age=3
X_pred = [[100000, 10], [50000, 3]]

# Task 1: Predict using sklearn KNN
y_pred, k_best = knn_sklearn(X, y, X_pred)
print(f"Predicted values using sklearn KNN: {y_pred}, Optimal k: {k_best}")

# Task 2: Predict using custom KNN with the same k
y_pred2 = knn(k_best, X, y, X_pred)
print(f"Predicted values using custom KNN: {y_pred2}")