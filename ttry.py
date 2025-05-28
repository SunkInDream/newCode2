import pandas as pd
from models_impute import impute
model_params = {
    'num_levels': 6,
    'kernel_size': 6,
    'dilation_c': 4,
    }
data = pd.read_csv('ICU_Charts/200003.csv')
cg = pd.read_csv('cgg.csv').values  
data_imputed = impute(data.values, cg, model_params=model_params)