from simplify_mesh import mesh_simplify
import numpy as np

v = np.random.rand(100, 3)
f = np.random.choice(range(100), (50, 3))
print(v.shape, f.shape)
v, f = mesh_simplify(v, f, 50)
print(v.shape, f.shape)
