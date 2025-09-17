import numpy as np
import matplotlib.pyplot as plt

from HHO_kernel import HHO_kernel 


poisson = HHO_kernel(np.linspace(0,1,17))
poisson.boundary_conditions = 'NN'

print(poisson.points)
print(poisson.spacing)
print(poisson.transmission_matrix.toarray())

poisson.solve_transmission(lambda x: np.pi**2*np.cos(np.pi*x))

print(f"Average of the discrete solution = {(poisson.average.toarray() @ poisson.solution_face.T)[0]}.")

plt.scatter(poisson.points, poisson.solution_face)
plt.scatter(poisson.points, np.cos(np.pi*poisson.points)+4)
plt.show()

