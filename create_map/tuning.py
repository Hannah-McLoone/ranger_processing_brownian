import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- Generate synthetic regression data ----
# 1000 samples, 20 features, 10 of which are informative

df_loaded = pd.read_csv('output/oban_features_speed_and_rank.csv')
X = df_loaded.iloc[:, :-1].values
y = df_loaded.iloc[:, -1].values


# Split into train/test sets
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---- Your RandomizedSearchCV code can run directly here ----



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
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

# Random search of parameters

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=10,
    cv=3,
    verbose=0,      # <- quiet
    random_state=42,
    n_jobs=-1
)



# Fit the random search model
rf_random.fit(train_features, train_labels)



print("Best parameters found: ")
print(rf_random.best_params_)

print("\nBest score from cross-validation: ")
print(rf_random.best_score_)
