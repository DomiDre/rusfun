import numpy as np
from scipy.optimize import leastsq
import lmfit
import time

rawdata = np.genfromtxt('./parabolaData.xye')
x = rawdata[:, 0]
y = rawdata[:, 1]
sy = rawdata[:, 2]

def linear(p, x):
  return p['a']*x**2 + p['b']*x + p['c']

def residuum(p, x, y, sy):
  return (y - linear(p, x))/sy

p = lmfit.Parameters()
p.add('a', 4)
p.add('b', 1)
p.add('c', 1)
t0 = time.time()
fit_result = lmfit.minimize(residuum, p, args=(x,y,sy))
exec_time = time.time() - t0
print(lmfit.fit_report(fit_result))
print(f"Execution time: {exec_time*1e3} ms")