import numpy as np
import pytest

from hho_1d import HHO_cell


# Define test polynomials to verify exactness
poly5 = lambda x, xL, xR: (x-xL)**5 - (x-xL)*(x-xR)
poly6 = lambda x, xL, xR: x*(x-xL)**5 - (x-xL)*(x-xR)
neg_dd_poly5 = lambda x, xL: -20*(x-xL)**3 + 2
neg_dd_poly6 = lambda x, xL: -20*x*(x-xL)**3 - 10*(x-xL)**4 + 2


@pytest.fixture
def cell_test():
    k = 5 # cell degree
    xL = 2.0 # left boundary
    xR = 2.5 # right boundary
    basis_type = "monomial"
    orthogonal_basis = False
    return HHO_cell(xL,xR,k,basis_type,orthonormal_basis=orthogonal_basis)


def grid_test(cell):
    return np.linspace(cell.x_left,cell.x_right,50)


def test_L2_projection(cell_test):
    # Since the test cell has polynomial degree 5, the L2 projection of 
    # 5th degree polynomials must be exact.
    poly = lambda x: poly5(x, cell_test.x_left, cell_test.x_right)
    # Compute projection
    xx = grid_test(cell_test)
    projection, _ = cell_test.evaluate_fun(
        xx, cell_test.compute_L2_projection(poly), "basis"
    )
    # Evaluate max error
    poly_value = poly(xx)
    err_max = np.max(np.abs( poly_value - projection ))
    assert err_max < 1e-13 * np.max(np.abs(poly_value))


def test_reconstruction(cell_test):
    # Since the test cell has polynomial degree 5, the reconstruction of
    # 6th degree polynomials must be exact.
    poly = lambda x: poly6(x, cell_test.x_left, cell_test.x_right)
    # Compute associated DOFs for the HHO space and recover the reconstruction
    dofs = np.concatenate((
        cell_test.compute_L2_projection(poly), 
        [poly(cell_test.x_left), poly(cell_test.x_right)]                   
    ))
    xx = grid_test(cell_test)
    reconstruction, _ = cell_test.cell_reconstruction.evaluate_fun(
        xx, cell_test.compute_reconstruction(dofs), "basis"
    )
    # Evaluate max error
    poly_value = poly(xx)
    err_max = np.max(np.abs( poly_value - reconstruction ))
    assert err_max < 1e-13 * np.max(np.abs(poly_value))


def execute_stabilization_test(cell, poly, neg_dd_poly):
    # Solve cell problem corresponding to this polynomial
    cell.solve(
        neg_dd_poly, poly(cell.x_left), poly(cell.x_right)
    )
    dofs = np.concatenate(
        (cell.solution, cell.solution_faces)
    )
    xx = grid_test(cell)
    solution, _ = cell.cell_reconstruction.evaluate_fun(
        xx, cell.compute_reconstruction(dofs), "basis"
    )
    # Evaluate max error
    poly_value = poly(xx)
    err_max = np.max(np.abs( poly_value - solution ))
    assert err_max < 1e-13 * np.max(np.abs(poly_value))


def test_cell_problem_stabilization_low_order(cell_test):
    # Since the test cell has polynomial degree 5, the cell problem with low-order stabilization
    # for 5th degree polynomials must be exact.
    cell_stab_low = HHO_cell(
        cell_test.x_left,
        cell_test.x_right,
        cell_test.degree,
        cell_test.basis_type,
        stabilization = 0 # deactivate higher-order stabilization
    )
    execute_stabilization_test(
        cell = cell_stab_low, 
        poly = lambda x: poly5(x, cell_test.x_left, cell_test.x_right),
        neg_dd_poly = lambda x: neg_dd_poly5(x, cell_test.x_left)
    )


def test_cell_problem_stabilization_high_order(cell_test):
    # Since the test cell has polynomial degree 5, the cell problem with high-order stabilization
    # for 6th degree polynomials must be exact.
    execute_stabilization_test(
        cell = cell_test, 
        poly = lambda x: poly6(x, cell_test.x_left, cell_test.x_right),
        neg_dd_poly = lambda x: neg_dd_poly6(x, cell_test.x_left)
    )
