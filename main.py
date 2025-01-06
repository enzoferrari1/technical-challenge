from src.dataset import CustomDataset
from src.models import CustomRandomForestModel

# Hiperparámetros
hparams = {
    'PARAM_GRID': {'n_estimators': [20, 30, 35, 40, 45, 50, 55, 60],
                'max_depth': [5, 6, 7, 10, 15, 20, 25, 30, 35, 40],},
    'LOSS': "MSE",
    'CROSS_VAL_N': 5,
    'TEST_SIZE':0.2,
    'RANDOM_STATE':42,
}

# Data
data = CustomDataset(csv_path="data/final_data.csv")

data_split = data.split_data(test_size=hparams.get('TEST_SIZE'), random_state=hparams.get('RANDOM_STATE'))

# Model
model = CustomRandomForestModel(param_grid=hparams.get('PARAM_GRID'), loss=hparams.get('LOSS'), cross_val_n=hparams.get('CROSS_VAL_N'), data=data_split)
model.build()

# Training Model
model.train()

# See metrics
model.get_metrics()

# Save model
model.save_model()



