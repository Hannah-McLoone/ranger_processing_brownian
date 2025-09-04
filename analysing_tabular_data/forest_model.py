import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def clean(df):
    df = df.drop(columns=['block_id'])

    df = df.drop(columns=[col for col in df.columns if col.startswith("humid")])
    df = df.replace(-np.inf, np.nan).dropna()


    return(df)



test = clean(pd.read_csv('oban_test.csv'))
train = clean(pd.read_csv('oban_train.csv'))



"""
#_________________________________________________________
#do this for all parks and combine

all_train = []
all_test = []

#for park in ['oban','afi']:
#    df = pd.read_csv(f'{park}_features_and_output.csv')
#    df['cover'] = df['cover'].map(lambda x: str(x))#!!!!!!!!!!!!!!!! turn into category, does this actually work?


# Combine all results at once
train = pd.concat(all_train, ignore_index=True)
test = pd.concat(all_test, ignore_index=True)


#________________________________________________________
"""



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


y_pred = [speed_train.mean()] * len(speed_test)

r2 = r2_score(speed_test, y_pred)
mae = mean_absolute_error(speed_test, y_pred)
rmse = np.sqrt(mean_squared_error(speed_test, y_pred))
print(r2,mae,rmse)

#____________________the machine learning___________________________

"""
model = RandomForestRegressor()
model.fit(X_train, speed_train)
speed_pred = model.predict(X_test)

r2 = r2_score(speed_test, speed_pred)
print(r2)
mae = mean_absolute_error(speed_test, speed_pred)
print(mae)
rmse = np.sqrt(mean_squared_error(speed_test, speed_pred))
print(rmse)
"""



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

models = {
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
        model.fit(X_train, speed_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(speed_test, y_pred)
        mae = mean_absolute_error(speed_test, y_pred)
        rmse = np.sqrt(mean_squared_error(speed_test, y_pred))

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


