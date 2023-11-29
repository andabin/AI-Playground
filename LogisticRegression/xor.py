from model import LinearModel
import matplotlib.pyplot as plt

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

linear_model = LinearModel(learning_rate=0.3)
linear_model.train(X, Y, 200)
