import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


exp_txts = [elm for elm in os.listdir() if ".txt" in elm]

experiments = []
epoch = 0
for txt_path in exp_txts:
    with open(txt_path, "r") as fp:
        text = fp.readlines()
        # epoch = int(text[0].split(" ")[-1])
        accuracy = float(text[1].split(" ")[-1])/100
        experiments.append([epoch,accuracy])
        epoch += 1

experiments = np.array(experiments)

fig, ax = plt.subplots(subplot_kw={'xlim': (0,40), 'ylim': (0,1)})
ax.scatter(experiments[:,0],experiments[:,1])
ax.plot(experiments[:,0],experiments[:,1])
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
plt.xlabel('Epoch', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)
ax.grid('on')