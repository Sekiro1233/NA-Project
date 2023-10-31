from scipy.optimize import minimize_scalar
import numpy as np

def f(x):
    return np.abs((x-2.0)*(x-2.4)*(x-2.6))

res = minimize_scalar(lambda x: -f(x), bounds=(2, 2.6), method='bounded')
print("The maximum value is:", -res.fun)
