
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mapping_dict = {
    10: 'A',
    20: 'B',
    30: 'C',
    40: 'D',
    50: 'E',
    60: 'F',
    70: 'G',
    80: 'H',
    90: 'I',
    95: 'J',
}

df_loaded = pd.read_csv('output/oban_features_speed_and_rank.csv')
df_loaded = pd.read_csv('analysing_tabular_data/feature_table.csv')
df_loaded['WorldCover'] = df_loaded['WorldCover'].map(lambda x: mapping_dict.get(x, x))#!!!!!!!!!!!!!!!!


# If you need X and y back:
X_loaded = df_loaded.iloc[:, :-2]#.values  # All columns except the last
y_loaded = df_loaded.iloc[:, -2]#.values   # Last column (speed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.2, random_state=42)

#scaler = StandardScaler() # MinMaxScaler for bounded or RobustScaler for heavy outliers
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


one_hot = ce.OneHotEncoder(cols=['WorldCover'])
X_train_scaled = one_hot.fit_transform(X_train)
X_test_scaled = one_hot.transform(X_test)



#____________________the machine learning___________________________


models = {
    #"Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": LinearSVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "MLP Neural Net": MLPRegressor(max_iter=1000),
    'Hist gradient boosting regressor': HistGradientBoostingRegressor(),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
    'extra trees': ExtraTreesRegressor(),
    'LGB': LGBMRegressor()
}

results = []

print("Evaluating regression models...\n")

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "R² Score": r2,
            "MAE": mae,
            "RMSE": rmse
        })

        print(f"✅ {name}")
        print(f"   R² Score : {r2:.4f}")
        print(f"   MAE      : {mae:.4f}")
        print(f"   RMSE     : {rmse:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"❌ {name} failed with error: {e}")
        print("-" * 40)

# Summary DataFrame
results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
print("\nSummary of all models:")
print(results_df)


