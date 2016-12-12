import numpy as np
a = [1,2,3,4,6]
print(a)
b = [[0,1] for _ in a]
print(b)
c = [[1,0] for _ in a]
print(c)
y = np.concatenate([b,c],0)
print(y)