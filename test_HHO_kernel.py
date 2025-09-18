import numpy as np
import matplotlib.pyplot as plt

from HHO_kernel import HHO_kernel 

def test_HHO(solver, f, ddf, bc, **bc_args):
    # Solve Poisson equation with HHO
    solver.boundary_conditions = bc
    solver.solve(ddf, **bc_args)
    # Compute max norm error of the transmission problem
    max_difference = np.max(np.abs(f(solver.points) - solver.solution_face))
    print(f"Max norm error of HHO at the faces = {max_difference}")
    # Plot the exact solution and the HHO approximation
    fig, ax = plt.subplots()
    xL = np.min(solver.points)
    xR = np.max(solver.points)
    xx = np.linspace(xL,xR,101)
    ax.plot(xx, 
            f(xx), 
            'k-', 
            linewidth=1, 
            label="Solution")
    solver.plot(ax)
    ax.tick_params(direction='in')
    ax.grid(True, 
            which='major', 
            linestyle='--', 
            linewidth=0.5, 
            color='lightgrey')
    plt.show()

poisson = HHO_kernel(np.linspace(0,1,17))

# Test 1 : f(x) = -4*(x^2 - x)
f = lambda x: -4*(x**2 - x) 
ddf = lambda x: 8
bc = 'DD'
test_HHO(poisson, f, ddf, bc)

# Test 2 : f(x) = 1 - x^3
f = lambda x: -x**3 + 1
ddf = lambda x: 6*x
bc = 'ND'
test_HHO(poisson, f, ddf, bc, bc_right=0)

# Test 3 : f(x) = 4 + cos(pi*x)
f = lambda x: 4 + np.cos(np.pi*x)
ddf = lambda x: np.pi**2 * np.cos(np.pi*x)
bc = 'NN'
test_HHO(poisson, f, ddf, bc, average=4)
