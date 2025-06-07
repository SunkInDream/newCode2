import pandas as pd
from models_impute import impute
from baseline import *
data = pd.read_csv('ICU_Charts/200003.csv').values
print(data)
data_imputed = tefn_impu(data)
print(data_imputed)
pd.DataFrame(data_imputed).to_csv('test.csv')