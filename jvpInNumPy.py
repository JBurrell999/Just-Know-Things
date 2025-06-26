import numpy as np

def f(x):
    return np.array([x[0] * x[1], np.sin(x[0])])

def jacobian(x):
    return np.array([
        [x[1], x[0]],
        [np.cos(x[0]), 0]
    ])

x = np.array([2.0, 3.0])
v = np.array([1.0, 0.0])  

J = jacobian(x)
jvp = J @ v

print("Jacobian-vector product:", jvp)
