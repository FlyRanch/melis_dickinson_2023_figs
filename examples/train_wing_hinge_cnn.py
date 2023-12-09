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
outliers = np.array([
    [18,0,100],
    [230,0,1000],
    [231,0,1000],
    [232,0,1000],
    [261,0,50],
    [338,0,1000],
    [345,100,150],
    [346,0,1000],
    [354,0,120]
    ])

net.create_dataset(outliers)
net.train_network(n_filters, curr_dir, save_history=True)

