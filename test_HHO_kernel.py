import numpy as np
import matplotlib.pyplot as plt

from HHO_kernel import HHO_kernel 

def test_HHO(solver, u, ddu, bc, **bc_args):
    # Solve Poisson equation with HHO
    solver.boundary_conditions = bc
    solver.solve(ddu, **bc_args)
    # Compute max norm error of the transmission problem
    max_difference = np.max(np.abs(u(solver.points) - solver.solution_face))
    print(f"Max norm error of HHO at the faces = {max_difference}")
    # Plot the exact solution and the HHO approximation
    fig, ax = plt.subplots()
    xL = np.min(solver.points)
    xR = np.max(solver.points)
    xx = np.linspace(xL,xR,101)
    ax.plot(xx, 
            u(xx), 
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
    ax.legend()
    plt.show()

poisson = HHO_kernel(np.linspace(0,1,9))

# Test 0 : u(x) = x
# The HHO reconstruction equals the exact solution
u = lambda x: x
ddu = lambda x: 0
bc = 'DD'
test_HHO(poisson, u, ddu, bc, bc_left=0, bc_right=1)


# Test 1 : u(x) = -4*(x^2 - x)
# Somewhat surprisingly, the reconstruction in this test case is continuous, but does not coincide with the face solutions at the nodes of the grid.
u = lambda x: -4*(x**2 - x) 
ddu = lambda x: 8
bc = 'DD'
test_HHO(poisson, u, ddu, bc)

# Test 2 : u(x) = 1 - x^3
u = lambda x: -x**3 + 1
ddu = lambda x: 6*x
bc = 'ND'
test_HHO(poisson, u, ddu, bc, bc_right=0)

# Test 3 : u(x) = 4 + cos(pi*x)
u = lambda x: 4 + np.cos(np.pi*x)
ddu = lambda x: np.pi**2 * np.cos(np.pi*x)
bc = 'NN'
test_HHO(poisson, u, ddu, bc, average=4)
