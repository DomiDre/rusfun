import numpy as np
import random

x = np.linspace(0, np.pi/2, 301)
A = 2.0
w = 4
phi = 0.42

y = A*np.cos(w*x - phi)
sig_y = 0.2*np.ones(len(y))#+0.05*np.abs(y)

randomized_y = []
for i in range(len(y)):
  randomized_y.append(random.gauss(y[i], sig_y[i]))
randomized_y = np.array(randomized_y)

np.savetxt('cosData.xye', np.c_[x,randomized_y,sig_y], fmt='%10.5f')