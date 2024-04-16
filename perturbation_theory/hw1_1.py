import numpy as np

def f1(eps,x):
    return eps*np.exp(x)

def f2(eps,x):
    return np.log(1/eps)+np.log(x)

# n = 100
# eps = 1/n
# #x0 = 1
# x0 = eps
# for i in range(1000):
#     x_next = f1(eps,x0)
#     print("x:", x0)
#     print("err:",np.exp(x_next)-x_next/eps)
#     x0 = x_next

n = 100000
eps = 1/n
x0 = 1000
for i in range(100):
    x_next = f2(eps,x0)
    print("x:", x0)
    print("err:",np.exp(x_next)-x_next/eps)
    x0 = x_next
