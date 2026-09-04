import joblib
import xgboost as xgb
import numpy as np

# Test: train, set feature_names, save, load
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
X_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, 20)

model = xgb.XGBClassifier()
model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
model.get_booster().feature_names = [f"feat_{i}" for i in range(10)]

# Simulate get_xgb_importance
booster = model.get_booster()
print("Before get_score - feature_names:", booster.feature_names)
booster.feature_names = [f"feat_{i}" for i in range(10)]  # Simulate get_xgb_importance setting names
print("After get_score - feature_names:", booster.feature_names)

# Save and load
joblib.dump(model, "test.joblib")
loaded = joblib.load("test.joblib")
print("After load:", loaded.get_booster().feature_names)