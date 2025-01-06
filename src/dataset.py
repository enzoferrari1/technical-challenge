import pandas as pd
from sklearn.model_selection import train_test_split

class CustomDataset():
    def __init__(self, csv_path="data/final_data.csv"):
        data = pd.read_csv(csv_path)
        data = data.replace({True: 1, False: 0})
        data = data.dropna()
        
        self.data = data

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=["Salary"])
        y = self.data['Salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test