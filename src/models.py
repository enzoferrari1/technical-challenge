from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import load, dump

from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

class CustomRandomForestModel():
    def __init__(self, param_grid=None, loss="MSE", cross_val_n=5, data=None):
        if param_grid == None:
            self.param_grid = {'n_estimators': [20, 30, 35, 40, 45, 50, 55, 60],
                'max_depth': [5, 6, 7, 10, 15, 20, 25, 30, 35, 40],}
        else:
            self.param_grid = param_grid
        if loss == "MSE":
            self.loss = mean_squared_error
        self.cross_val_n = cross_val_n
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.grid_search = None
        self.best_rf_regressor = None
        
    def build(self):
        # Initialize the model
        rf = RandomForestRegressor()

        # Scorer for determining the metric to optimize (negative mean squared error)
        scorer = make_scorer(self.loss, greater_is_better=False)

        # GridSearchCV for cross-validation
        grid_search = GridSearchCV(estimator=rf, param_grid=self.param_grid, scoring=scorer, cv=self.cross_val_n)
        self.grid_search = grid_search

    def train(self):
        print("Training model...")
        self.grid_search.fit(self.X_train, self.y_train)

        # Get the best estimator from the grid search
        best_rf_regressor = self.grid_search.best_estimator_
        self.best_rf_regressor = best_rf_regressor
        print("Finished")
        
    def inference(self, input):
        pred = self.best_rf_regressor.predict(input)
        return pred

    def get_metrics(self):
        # Predictions on the test set
        self.y_pred = self.inference(self.X_test)

        # Best parameters found
        print(f"Best number of estimators: {self.grid_search.best_params_['n_estimators']}")
        print(f"Best maximum tree depth: {self.grid_search.best_params_['max_depth']}")
        print("Results for the best model:")

        # Evaluation (Mean Squared Error)
        print("R2 score:", r2_score(self.y_test, self.y_pred))
        print("MAE:", mean_absolute_error(self.y_test, self.y_pred))

    def save_model(self, path="saved/best_rf.joblib"):
        dump(self.best_rf_regressor, path)

    def load_model(self, path="saved/best_rf.joblib"):
        self.best_rf_regressor = load(path)
