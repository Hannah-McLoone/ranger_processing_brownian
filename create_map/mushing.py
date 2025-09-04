import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
def clean(df):
    df = df.drop(columns=['block_id'])

    df = df.drop(columns=[col for col in df.columns if col.startswith("humid")])
    df = df.replace(-np.inf, np.nan).dropna()


    return(df)

# assuming df is your dataframe
df_train = clean(pd.read_csv('oban_train.csv'))
df_test = clean(pd.read_csv('oban_test.csv'))


def weighted_avg_speed(group):
    return (group["speed"] * group["rank"]).sum() / group["rank"].sum()

# group by lon & lat
df_train = (
    df_train.groupby(["lons", "lats"], as_index=False)
      .apply(lambda g: pd.Series({
          "speed": weighted_avg_speed(g),
          "rank": g["rank"].sum(),
          **{col: g[col].iloc[0] for col in df_train.columns if col not in ["lons", "lats", "speed", "rank"]}
      }))
)



df_test = (
    df_test.groupby(["lons", "lats"], as_index=False)
      .apply(lambda g: pd.Series({
          "speed": weighted_avg_speed(g),
          "rank": g["rank"].sum(),
          **{col: g[col].iloc[0] for col in df_test.columns if col not in ["lons", "lats", "speed", "rank"]}
      }))
)

# reset index just in case
df_test.reset_index(drop=True, inplace=True)
df_test = df_test.drop(columns=['temperature', 'precipitation', 'lons', 'lats', 'park', 'rank'])

df_train.reset_index(drop=True, inplace=True)
df_train = df_train.drop(columns=['temperature', 'precipitation', 'lons', 'lats', 'park', 'rank'])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import category_encoders as ce

# Features and target
X_train = df_train.drop(columns=['speed'])
y_train = df_train['speed']

X_test = df_test.drop(columns=['speed'])
y_test = df_test['speed']
# Split into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
one_hot = ce.OneHotEncoder(cols=['cover'])
X_train = one_hot.fit_transform(X_train)
X_test = one_hot.transform(X_test)


# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))


"""

# Cap rank at 10,000
merged['rank_capped'] = merged['rank'].clip(upper=10000)

# Create scatter map
fig = px.scatter_mapbox(
    merged,
    lat='lats',
    lon='lons',
    color='rank_capped',          # Color by capped rank
    color_continuous_scale='Viridis',
    size_max=15,
    zoom=3,
    mapbox_style="carto-positron"
)

fig.show()
"""