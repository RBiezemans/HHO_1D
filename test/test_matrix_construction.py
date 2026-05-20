import numpy as np
import pytest 

from hho_1d import HHO_cell


@pytest.fixture 
def cell_test():
    cell = HHO_cell(
        x_left = -1,
        x_right = 1,
        degree = 3, 
        basis = "Monomial",
        orthonormal_basis = False
    )
    return cell


@pytest.fixture 
def cell_test_orthonormal():
    cell = HHO_cell(
        x_left = -1,
        x_right = 1,
        degree = 3, 
        basis = "Monomial",
        orthonormal_basis = True
    )
    return cell


def test_mass_matrix(cell_test):
    # The correct mass matrix on (-1,1) is given by
    # M_{k,l} = \int_{-1}^1 x^{k+l} = 2/(k+l+1) if (k+l) is even, 0 otherwise.
    correct_matrix = np.array([
        [2.0/(k+l+1) if not (k+l)%2 else 0 for k in range(cell_test.degree+1)]
        for l in range(cell_test.degree+1)
    ])
    np.testing.assert_allclose(
        cell_test.mass_matrix,
        correct_matrix,
        rtol = 1e-10,
        atol = 1e-10
    )


def test_stiffness_matrix(cell_test):
    # The correct stiffness matrix on (-1,1) is given by
    # M_{k,l} = k*l*\int_{-1}^1 x^{k+l-2} = 2*k*l/(k+l-1) if (k+l) is even and k>0, 0 otherwise.
    correct_matrix = np.array([
        [2.0*k*l/(k+l-1) if (not (k+l)%2) and k>0 else 0 for k in range(cell_test.degree+1)]
        for l in range(cell_test.degree+1)
    ])
    np.testing.assert_allclose(
        cell_test.stiffness_matrix,
        correct_matrix,
        rtol = 1e-10,
        atol = 1e-10
    )


def test_mass_matrix_orthonormal(cell_test_orthonormal):
    # The mass matrix msust equal the identity matrix.
    np.testing.assert_allclose(
        cell_test_orthonormal.mass_matrix,
        np.eye(cell_test_orthonormal.degree+1),
        rtol = 1e-10,
        atol = 1e-10
    )


def test_stiffness_matrix_orthonormal(cell_test_orthonormal):
    # The correct stiffness matrix can be computed with known results 
    # for Legendre polynomials.
    correct_matrix = np.array(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 3.0, 0.0, np.sqrt(21.0)],
         [0.0, 0.0, 15.0, 0.0],
         [0.0, np.sqrt(21.0), 0.0, 42]]
    )
    np.testing.assert_allclose(
        cell_test_orthonormal.stiffness_matrix,
        correct_matrix,
        rtol = 1e-10,
        atol = 1e-10
    )
