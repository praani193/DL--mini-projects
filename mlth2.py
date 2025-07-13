import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = "C:/Users/Lenovo/Downloads/StudentPerformanceFactors.csv"  # Update with your path
data = pd.read_csv(file_path)

# Split data into features and target
X = data.drop("Exam_Score", axis=1)
y = data["Exam_Score"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Define preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines into a preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Define models with basic hyperparameters
ridge = Ridge()
lasso = Lasso()
gb = GradientBoostingRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
svr = SVR()

# Hyperparameter tuning using RandomizedSearchCV
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}  # 4 options
lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}  # 4 options
gb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
svr_params = {'C': [0.1, 1.0, 10], 'epsilon': [0.01, 0.1, 1]}

# Create RandomizedSearchCV for each model
ridge_search = RandomizedSearchCV(ridge, ridge_params, cv=5, n_iter=4, random_state=42)
lasso_search = RandomizedSearchCV(lasso, lasso_params, cv=5, n_iter=4, random_state=42)
gb_search = RandomizedSearchCV(gb, gb_params, cv=5, n_iter=10, random_state=42)
rf_search = RandomizedSearchCV(rf, rf_params, cv=5, n_iter=10, random_state=42)
svr_search = RandomizedSearchCV(svr, svr_params, cv=5, n_iter=10, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ridge_search)
])

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lasso_search)
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', gb_search)
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_search)
])

svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', svr_search)
])

# Fit the models
ridge_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
svr_pipeline.fit(X_train, y_train)

# Make predictions and evaluate the models
models = {
    "Ridge": ridge_pipeline,
    "Lasso": lasso_pipeline,
    "Gradient Boosting": gb_pipeline,
    "Random Forest": rf_pipeline,
    "SVR": svr_pipeline
}

for name, pipeline in models.items():
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MAE: {mae:.3f}, R²: {r2:.3f}")

# Ensemble Voting Regressor combining the best models
voting_regressor = VotingRegressor(estimators=[
    ('ridge', ridge_search.best_estimator_),
    ('gb', gb_search.best_estimator_),
    ('rf', rf_search.best_estimator_)
])
voting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', voting_regressor)
])

# Fit and evaluate Voting Regressor
voting_pipeline.fit(X_train, y_train)
voting_pred = voting_pipeline.predict(X_test)
voting_mae = mean_absolute_error(y_test, voting_pred)
voting_r2 = r2_score(y_test, voting_pred)
print(f"Voting Regressor - MAE: {voting_mae:.3f}, R²: {voting_r2:.3f}")

# Visualize feature importance for tree-based models (Random Forest or Gradient Boosting)
importances = gb_search.best_estimator_.feature_importances_
feature_names = np.concatenate([numerical_cols, gb_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()])
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx])
plt.yticks(range(len(importances)), feature_names[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance")
plt.show()

# Plot residuals for ensemble model
plt.figure(figsize=(15, 5))
plt.scatter(y_test, voting_pred - y_test, alpha=0.5, color='g')
plt.axhline(0, color='r', linestyle='--')
plt.title('Voting Regressor Residuals')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Residuals')
plt.show()
