import torch
import matplotlib.pyplot as plt
from data_loader import get_data_loaders

# Define the root path of the dataset
root_path = "/data"
print('MAIN-1:Root Path:', root_path);

# Define hyperparameters
hparams = {
    'batch_size': 64,
    'num_epochs': 10,
    'test_batch_size': 64,
    'learning_rate': 1e-4,
}
print('MAIN-2:hparams:', hparams);

# Load data using the data loader function, divide in train and test
train_loader, test_loader = get_data_loaders(root_path, hparams['batch_size'], hparams['test_batch_size'])

print('MAIN-3:train_loader and test_loader load');

