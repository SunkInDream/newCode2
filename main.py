from models_dataset import *
from models_CAUSAL import *
dataset = MyDataset('./data')
print(len(dataset))
print(dataset[0]['original'])
print(dataset[0]['mask'])
print(dataset[0]['initial_filled'])