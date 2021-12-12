import pickle

import matplotlib.pyplot as plt
from parameters import PIK_plot_data

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


with open(PIK_plot_data, "rb") as f:
    data = pickle.load(f)
train_loss = data['train_loss']
train_acc = data['train_acc']
test_acc = data['test_acc']

plt.plot(train_loss, label="Train Loss")
plt.xlabel(" Iteration ")
plt.ylabel("Loss value")
plt.show()

plt.plot(train_acc, label="Train Acc")
plt.plot(test_acc, label="Test Acc")
plt.xlabel(" Iteration ")
plt.ylabel("Accuracy value")
plt.legend(loc="upper left")
plt.show()
