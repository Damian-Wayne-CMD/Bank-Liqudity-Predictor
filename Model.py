import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

liquidity_data = pd.read_csv('/Users/brb/Documents/Coding Projects/ML Model - Liquidity Predictor/liquidity_data.csv')

### Exploratory data analysis
liquidity_data.head(5)
liquidity_data.describe()
#Examine target distribution
sns.histplot(liquidity_data["available_liquidity"])
#Missing Values check
liquidity_data.isnull().sum()

### Training & Evaluation
target_y = liquidity_data["available_liquidity"]
inputs_x = liquidity_data.drop("available_liquidity", axis=1)
target_y.head(2)
inputs_x.head(2)

results = train_test_split(inputs_x,target_y, test_size=0.2, random_state=1)

# Type Test Verification
print(type(results))
print(len(results))
print("Divider ---- Line")

for item in results: 
    print(item.shape)

# Double-check & unpack results to ensure proper set-up
x_train, x_test, y_train, y_test = results
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train.head(1)
x_test.head(1)
y_train.head(1)
y_test.head(1)

### Modeling
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Create pipelines dictionary with model pipelines for Lasso & Ridge
pipelines = {
    "lasso" : Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(random_state=1))
    ]), 
    "ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=1))
    ]),
     "enet": Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(random_state=1))
        ]),
    "rf" : Pipeline([
        ("rf", RandomForestRegressor(random_state=1))
    ]),
    "gb" : Pipeline([
            ("gb", GradientBoostingRegressor(random_state=1))
        ]),

}
#Verify pipelines are properly defined
for key, value in pipelines.items():
    print(key, type(value))

### Hyperparameter Tuning
lasso_hyperparameters = { 
    'lasso__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
}

ridge_hyperparameters = { 
    'ridge__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
}

enet_hyperparameters = {
    'enet__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5],
    'enet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

rf_hyperparameters = { 
    'rf__n_estimators' : [100,200],
    'rf__max_features' : [0.1, 0.3, 0.5]
}

gb_hyperparameters = {
    'gb__n_estimators': [100,200], 
    'gb__learning_rate' : [0.05, 0.1, 0.2],
    'gb__max_depth' : [1,3,5]
}

hyperparameter_grids = {
    'lasso': lasso_hyperparameters, 
    'ridge': ridge_hyperparameters,
    'enet': enet_hyperparameters,
    'rf': rf_hyperparameters,
    'gb': gb_hyperparameters
}

#Run loop to ensure dictionary is set up correctly
for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameter_grids:
        if type(hyperparameter_grids[key]) is dict:
            print(key, 'was found and it is a grid.')
        else: 
            print(key, "was found, but it is not a grid")
    else: 
        print(key, "was NOT found")


### Cross Validation
from sklearn.model_selection import GridSearchCV

models = {}

for key in pipelines.keys() : 
    models[key] = GridSearchCV(pipelines[key], hyperparameter_grids[key], cv=5)

    models.keys()

for key in models.keys() : 
    models[key].fit(x_train, y_train)
    print(key, 'is trained & tuned')

### Selecting a Winning Model
from sklearn.metrics import r2_score, mean_absolute_error

for key in models.keys(): 
    preds = models[key].predict(x_test)
    print(key)
    print('R-Squared: ', round(r2_score(y_test, preds), 3))
    print('MAE: ', round(mean_absolute_error(y_test, preds), 3))
    print('---Divider---')

#Make predictions with test data
preds = models['gb'].predict(x_test)
plt.scatter(preds, y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
