# HHO_1D
*1D solver for the Poisson equation by the HHO method.*

## Features of the project
The solver implements:
* Arbitrary polynomial degree for the cell unknowns (but for high-order approximations make sure to use the Legendre polynomials as the basis functions on the cell).
* Arbitrary Dirichlet conditions.
* Homogeneous Neumann conditions.

## Getting started
You may start by having a look at the [getting_started](getting_started.ipynb) Jupyter notebook, which explains the interface to the solver.

More examples and validation of the solver are included in the Jupter notebooks [validation_lowest_order](validation_lowest_order.ipynb) and [validation_higher_order](validation_higher_order.ipynb).