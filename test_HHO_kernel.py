import numpy as np
import matplotlib.pyplot as plt

from HHO_kernel import HHO_kernel 


poisson = HHO_kernel(np.linspace(0,1,17))


# print(poisson.points)
# print(poisson.spacing)
# print(poisson.transmission_matrix.toarray())

# poisson.boundary_conditions = 'NN'
# poisson.solve_transmission(lambda x: np.pi**2*np.cos(np.pi*x), average=4)

# poisson.boundary_conditions = 'DD'
# poisson.solve_transmission(lambda x: np.pi**2*np.cos(np.pi*x), bc_left = 1.0, bc_right = -1.0)

poisson.boundary_conditions = 'DD'
poisson.solve_transmission(lambda x: 2, bc_left=1, bc_right=0)

plt.scatter(poisson.points, poisson.solution_face)
# plt.scatter(poisson.points, np.cos(np.pi*poisson.points)+4)
f = lambda x: -x*(x-1) - x + 1
plt.scatter(poisson.points, f(poisson.points))
plt.show()




