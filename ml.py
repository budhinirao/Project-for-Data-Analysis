# ml.py
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def train_extra_trees(X, y):
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create the model
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)

    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the model
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    cv_score = -np.mean(cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=5))
    r2 = r2_score(y_test, pred)

    return model, mse, cv_score, r2