import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("AB_NYC_2019.csv")

# 2. Remove irrelevant columns
columns_to_drop = [
    "id",
    "name",
    "host_id",
    "host_name",
    "last_review"
]
df = df.drop(columns=columns_to_drop)

# 3. Fill missing reviews_per_month
df["reviews_per_month"] = df["reviews_per_month"].fillna(
    df["reviews_per_month"].median()
)

# 4. Remove invalid/outlier prices
df = df[df["price"] > 0].copy()
price_99 = df["price"].quantile(0.99)
df = df[df["price"] <= price_99].copy()

# 5. Remove extreme minimum_nights
minimum_nights_99 = df["minimum_nights"].quantile(0.99)
df = df[df["minimum_nights"] <= minimum_nights_99].copy()

# 6. Feature engineering
df["reviews_per_availability"] = (
    df["number_of_reviews"] / (df["availability_365"] + 1)
)

df["host_listing_density"] = (
    df["calculated_host_listings_count"] /
    (df["availability_365"] + 1)
)

# 7. Features and target
features = [
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "reviews_per_availability",
    "host_listing_density"
]

X = df[features].copy()
y = df["price"].copy()

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 9. Preprocessing
numeric_features = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "reviews_per_availability",
    "host_listing_density"
]

categorical_features = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 10. Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])

# 11. Hyperparameter search
param_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5]
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training model... This may take a few minutes.")
random_search.fit(X_train, y_train)

# 12. Evaluate
final_model = random_search.best_estimator_

train_pred = final_model.predict(X_train)
test_pred = final_model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("\nBEST HYPERPARAMETERS")
print(random_search.best_params_)

print("\nFINAL MODEL PERFORMANCE")
print("=======================")
print("Train R² :", round(train_r2, 4))
print("Test R²  :", round(test_r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))
print("R² Gap   :", round(train_r2 - test_r2, 4))

# 13. Save model using the CURRENT scikit-learn 1.9.0 environment
joblib.dump(final_model, "airbnb_price_model.pkl")

print("\nSUCCESS!")
print("New model saved as: airbnb_price_model.pkl")

# 14. Test the newly saved model
loaded_model = joblib.load("airbnb_price_model.pkl")

new_input = pd.DataFrame({
    "neighbourhood_group": ["Manhattan"],
    "neighbourhood": ["Midtown"],
    "latitude": [40.7549],
    "longitude": [-73.9840],
    "room_type": ["Entire home/apt"],
    "minimum_nights": [3],
    "number_of_reviews": [50],
    "reviews_per_month": [2.5],
    "calculated_host_listings_count": [2],
    "availability_365": [200],
    "reviews_per_availability": [50 / 201],
    "host_listing_density": [2 / 201]
})

prediction = loaded_model.predict(new_input)[0]

print("Test prediction: $", round(prediction, 2))
