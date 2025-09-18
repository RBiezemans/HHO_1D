import numpy as np
import matplotlib.pyplot as plt

from HHO_kernel import HHO_kernel 


poisson = HHO_kernel(np.linspace(0,1,17))

poisson.boundary_conditions = 'NN'
poisson.solve_transmission(lambda x: np.pi**2*np.cos(np.pi*x), average=4)
f = lambda x: np.cos(np.pi*x)+4
max_difference = np.max(np.abs(f(poisson.points) - poisson.solution_face))
print(f"Max norm difference = {max_difference}")
plt.scatter(poisson.points, poisson.solution_face)
plt.scatter(poisson.points, f(poisson.points))
plt.show()

poisson.boundary_conditions = 'ND'
poisson.solve_transmission(lambda x: 2, bc_right=0)
plt.scatter(poisson.points, poisson.solution_face)
f = lambda x: -x*(x-1) - x + 1
max_difference = np.max(np.abs(f(poisson.points) - poisson.solution_face))
print(f"Max norm difference = {max_difference}")
plt.scatter(poisson.points, f(poisson.points))
plt.show()




