import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from wing_hinge_cnn import WingHingeCNN

n_filters = 9
curr_dir = pathlib.Path().cwd()
dataset_dir = curr_dir / 'main_muscle_and_wing_data.h5'

net = WingHingeCNN()
net.load_dataset(dataset_dir)
outliers = -np.ones((1,1)) 
net.create_dataset(outliers)
net.train_network(n_filters, curr_dir, save_history=True)

