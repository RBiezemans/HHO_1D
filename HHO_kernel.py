import numpy as np
import scipy.sparse as sp

class HHO_kernel:
    """
    Object with the functionality to build the matrix and vector of the discrete linear system.

    Attributes
    ----------
        points : ndarray
            Vertices of the discrete domain.
        spacing : ndarray
            Spacing[i] = distance between points[i+1] and points[i].
        nb_face_unkowns : int
            Number of face unknowns.
        cell_degree : int
            Polynomial degree of the cell unknowns.
        boundary_conditions : str
            Description of the boundary conditions at both ends of the domain.
            Format 'XX' with X either N or D, and the left (resp. right) symbol denotes the BC at the left (resp. right)
            'N' -> Neumann 
            'D' -> Dirichlet
            Note that 'NN' requires specification of the average in the sovler
        transmission_matrix : scipy.sparse.spmatrix
            Matrix associated to the global transmission problem.
        transmission_average : scipy.sparse.spmatrix
            Row vector representing the average operator in the global transmission problem.
        transmission_average : scipy.sparse.spmatrix
            Matrix for the global problem including boundary conditions through Lagrange multipliers.
        solution_face : ndarray
            Solution vector of the transmission problem.
    """

    def __init__(self, x, degree=0):
        """
        Initialize HHO kernel: build discrete linear system (global and independent local problems).

        Args:
            x : ndarray
                vertices of the discrete domain
            degree : int
                polynomial degree of the cell unknowns
        """
        self.points = x
        self.spacing = x[1:] - x[0:-1]

        self.nb_face_unknowns = len(self.points)
        self.cell_degree = degree

        # Initialize quantities related to the linear system to None
        self.transmission_matrix = None 
        self.transmission_average = None
        self._transmission_system = None
        self._boundary_conditions = None

        self.build_transmission_matrix()
        self.build_transmission_average()

    @property 
    def boundary_conditions(self):
        return self._boundary_conditions
    
    @boundary_conditions.setter 
    def boundary_conditions(self, bc):
        if self.boundary_conditions != bc:
            self._boundary_conditions = bc
            self.build_transmission_system()

    @property 
    def transmission_system(self):
        """
        Matrix for the global problem including boundary conditions through Lagrange multipliers.

        The LU decomposition is actually stored, in SciPy SparseLU format (scipy.sparse.linalg.splu)
        It is automatically recomputed when the boundary conditions are modified.
        """
        if self._transmission_system is None:
            self.build_transmission_system()
        return self._transmission_system

    def build_transmission_system(self):
        match self.boundary_conditions:
            case 'NN':
                A = sp.block_array([[self.transmission_matrix, self.average.T], [self.average, None]], format='csc')
            case _:
                raise ValueError(f"Unsupported boundary conditions [{self.bc}] for global transmission problem.")
        print("...performing LU...")
        self._transmission_system = sp.linalg.splu(A)

    def build_transmission_matrix(self):
        """Build discrete linear system of the global transmission problem with Neumann conditions."""
        # Compute the three nonzero diagonals
        upper_diag = -1/self.spacing 
        lower_diag = upper_diag
        diag = np.zeros(self.nb_face_unknowns)
        diag[:-1] = 1/self.spacing 
        diag[1:] += 1/self.spacing
        # Save the linear system as a sparse matrix
        self.transmission_matrix = sp.diags_array([upper_diag, diag, lower_diag], offsets=[+1, 0, -1])

    def build_transmission_average(self):
        """Build row vector corresponding to the discrete average over the conforming P1 space."""
        # Compute vector corresponding to the whole integral
        average = np.zeros(self.nb_face_unknowns)
        average[0:-1] = self.spacing/2.0
        average[1:] += self.spacing/2.0
        # Divide by length of the computational domain
        length = self.points[-1] - self.points[0]
        average = average/length
        # Make it a row vector
        average = average[np.newaxis]
        # Save in sparse format for compatibilty with sp.block_array
        self.average = sp.coo_array(average)
        
    def build_transmission_rhs(self, f):
        """
        Build discrete right-hand side associated to the function f.

        Parameters
        ----------
            f : callable
                Source term of the Poisson equation.
                Should accept an ndarray and return an ndarray of the same size 
        """
        self.transmission_rhs = np.zeros(self.nb_face_unknowns)
        f_half_average = 0.5 * self.spacing * f((self.points[0:-1] + self.points[1:])/2.0)
        self.transmission_rhs[0:-1] = f_half_average 
        self.transmission_rhs[1:] = self.transmission_rhs[1:] + f_half_average

    def solve_transmission(self, f):
        """
        Solve the global transmission problem and store the solution in self.solution_face.

        Parameters
        ----------
            f : callable
                Source term of the Poisson equation.
                Should accept an ndarray and return an ndarray of the same size 
        """
        self.build_transmission_rhs(f)
        rhs = np.concatenate((self.transmission_rhs, np.array([4])))
        sol = self.transmission_system.solve(rhs)
        self.solution_face = sol[0:self.nb_face_unknowns]


    