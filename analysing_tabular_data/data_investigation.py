import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

"""
simple data science:
Simply plotting the relationships between speed and features
seeing correlation between featues - 
since there is strong correlation between population and elevation,
then regularisation (like ridge) might be useful

trying ridge and lasso with a few values for alpha, just to see.



"""
# correlation heatmaps

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

# Map values using the dictionary

df = pd.read_csv('temp_table.csv')
#df['WorldCover'] = df['WorldCover'].map(lambda x: mapping_dict.get(x, x))#!!!!!!!!!!!!!!!!
correlation_matrix = df.corr()

# Plot heatmap


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
"""


# If you need X and y back:
plt.scatter(df_loaded['population'], df_loaded['speed'], s=1, alpha = 0.003)
plt.xlabel("population")
plt.ylabel("speed")
plt.show()

plt.scatter(df_loaded['elevation'], df_loaded['speed'], s=1, alpha = 0.003)
plt.xlabel("elevation")
plt.ylabel("speed")
plt.show()


plt.scatter(df_loaded['forest'], df_loaded['speed'], s=1, alpha = 0.003)
plt.xlabel("forest")
plt.ylabel("speed")
plt.show()
"""



# Load data
df_loaded = pd.read_csv('features_and_speed.csv')
X = df_loaded.iloc[:, :-1].values
y = df_loaded.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() # MinMaxScaler for bounded or RobustScaler for heavy outliers

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Alpha values to test
ridgeList = []
lasso_list = []
alphas = np.logspace(-1, 4, 50)
for alpha in alphas:

    # Ridge
    ridge = Ridge(alpha=alpha, max_iter=10000)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)


    ridgeList.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    lasso_list.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))


plt.plot(alphas,ridgeList, label='ridge')
plt.plot(alphas,lasso_list, label='lasso')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.legend()
plt.show()
