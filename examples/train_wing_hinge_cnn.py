import pickle
import pathlib
import matplotlib.pyplot as plt
from wing_hinge_cnn import WingHingeCNN

dataset_dir = pathlib.Path().cwd() / 'dataset_melis_et_al_2023.h5'

net = WingHingeCNN()
net.load_dataset(dataset_dir)
res = net.train_network()

with open('history.pkl', 'wb') as f:
    pickle.dump(res, f)

plt.show()
