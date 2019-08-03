import numpy as np
import random

x = np.linspace(0, 10, 101)
a = 4.2
b = 0.666
c = -0.23

y = a*x**2 + b*x + c
sig_y = 0.05*np.abs(y)

randomized_y = []
for i in range(len(y)):
  randomized_y.append(random.gauss(y[i], sig_y[i]))
randomized_y = np.array(randomized_y)

np.savetxt('parabolaData.xye', np.c_[x,randomized_y,sig_y], fmt='%10.5f')