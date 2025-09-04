import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import category_encoders as ce
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def clean(df):
    df = df.drop(columns=['block_id'])

    df = df.drop(columns=[col for col in df.columns if col.startswith("humid")])
    df = df.replace(-np.inf, np.nan).dropna()


    return(df)



test = clean(pd.read_csv('/maps/hm708/final/oban_test.csv'))
train = clean(pd.read_csv('/maps/hm708/final/oban_train.csv'))

speed_train = train['speed']
rank_train = train['rank']
X_train = train.drop(columns=['speed', 'rank'])


speed_test = test['speed']
rank_test = test['rank']
X_test = test.drop(columns=['speed', 'rank'])


one_hot = ce.OneHotEncoder(cols=['cover','park' ])
X_train = one_hot.fit_transform(X_train)
X_test = one_hot.transform(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Define random grid (same as before)
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 20, 50, 100]
min_samples_leaf = [1, 2, 4, 10, 20, 50]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# Base model
rf = RandomForestRegressor(random_state=42)

# Random search with RMSE scoring
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=50,
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=10,
    scoring='neg_root_mean_squared_error'
)

# Fit model
rf_random.fit(X_train, speed_train)

results_df = pd.DataFrame(rf_random.cv_results_)

# Keep only useful columns
results_df = results_df[
    [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].copy()

# Convert negative RMSE back to positive
results_df["mean_test_score"] = -results_df["mean_test_score"]
results_df["std_test_score"] = results_df["std_test_score"].abs()

# Save to CSV
results_df.to_csv("tuned.csv", index=False)
