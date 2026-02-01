import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv("kc_house_data.csv")

# Select useful numeric features
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "grade"
]

X = data[features]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("RMSE:", rmse)



