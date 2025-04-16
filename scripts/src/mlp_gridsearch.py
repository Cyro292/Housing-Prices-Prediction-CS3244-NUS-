from lib.utils import get_train_split_data, load_all_resale_data, get_cleaned_normalized_data
from lib.eval import get_regression_metrics

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIDDEN_LAYER_SIZES = (64, 32)
MAX_ITER = 200
X, y = load_all_resale_data()

X, y = get_cleaned_normalized_data(X, y)

# Split data
print(f"Splitting data with test size {TEST_SIZE}...")
X_train, X_test, y_train, y_test = get_train_split_data(X, y, TEST_SIZE)
# Initialize model
print("Initializing MLPRegressor...")
model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)
# Train model
print("Training neural network...")
model.fit(X_train, y_train)

# Hyperparameter tuning
# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.0005, 0.01],
    'solver': ['adam', 'sgd']  # 'lbfgs' can also be included if desired
}
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
# Fit grid search on training data
grid_search.fit(X_train, y_train)
# Display best parameters
print("Best parameters found: ", grid_search.best_params_)
print("Best MSE: ", -grid_search.best_score_)