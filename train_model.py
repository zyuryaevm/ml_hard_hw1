import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import pickle

data = pd.read_csv('WineQT.csv')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def run_experiment(model_name, model, model_params):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        model_predict = model.predict(X_test)

        r2 = r2_score(y_test, model_predict)
        mse = mean_squared_error(y_test, model_predict)

        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mse', mse)

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        mlflow.sklearn.log_model(model, 'model')

        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)


mlflow.set_experiment('First Experiment - Random Forests')

model1 = RandomForestRegressor(random_state=42)
model1_params = {'model_type': 'RandomForestRegressor',
                 'random_state': 42}
run_experiment('RandomForestRegressor1', model1, model1_params)

model2 = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=42)
model2_params = {'model_type': 'RandomForestRegressor',
                 'n_estimators': 100,
                 'max_depth': 100,
                 'random_state': 42}
run_experiment('RandomForestRegressor2', model2, model2_params)


mlflow.set_experiment('Second Experiment - Linear Regression')

model3 = LinearRegression()
model3_params = {'model_type': 'Linear Regression'}
run_experiment('LinearRegression', model3, model3_params)

model4 = ElasticNet(alpha=3)
model4_params = {'model_type': 'Elastic Net',
                 'alpha': 3}
run_experiment('ElasticNet', model4, model4_params)


mlflow.set_experiment('Third Experiment - SVM')

model5 = SVR(kernel='rbf')
model5_params = {'model_type': 'SVR',
                 'kernel': 'rbf'}
run_experiment('SVR', model5, model5_params)