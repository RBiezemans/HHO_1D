import numpy as np
import scipy.sparse as sp
import warnings 

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
            Format 'XX' with X either N (Neumann) or D (Dirichlet), and the left (resp. right) symbol denotes the BC at the left (resp. right).
            Note that 'NN' requires specification of the average in the solver.
        transmission_matrix : scipy.sparse.spmatrix
            Matrix associated to the global transmission problem.
        transmission_average : scipy.sparse.spmatrix
            Row vector representing the average operator in the global transmission problem.
        transmission_average : scipy.sparse.spmatrix
            Matrix for the global problem including boundary conditions through Lagrange multipliers.
        solution_face : ndarray
            Solution vector of the transmission problem.

    Methods
    -------
        solve_transmission(f, bc_left=None, bc_right=None, average=None)
            Solve the global transmission problem with source term f.
            Non-homogeneous Dirichlet boundary conditions are set by specifying bc_left, bc_right.
            In case of pure Neumann conditions, the average of the solution is set by specifying average.
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

        # Initialize quantities related to the linear system to unset state
        self._transmission_matrix = None 
        self._transmission_average = None
        self._transmission_system = None
        self._boundary_conditions = None

    @property 
    def boundary_conditions(self):
        """
        Description of the boundary conditions at both ends of the domain.
        
        The system of the global problem is recomputed when the boundary conditions are changed.
        """
        return self._boundary_conditions
    
    @boundary_conditions.setter 
    def boundary_conditions(self, bc):
        if self.boundary_conditions != bc:
            self._boundary_conditions = bc
            self._transmission_system = None

    @property 
    def transmission_system(self):
        """
        Matrix for the global problem including boundary conditions through Lagrange multipliers.

        The LU decomposition is actually stored, in SciPy SparseLU format (scipy.sparse.linalg.splu)
        It is automatically recomputed when the boundary conditions are modified.
        """
        if self._transmission_system is None:
            self._build_transmission_system()
        return self._transmission_system

    def _build_transmission_system(self):
        """
        Assemble the linear system for the transmission matrix.
        
        The matrix of the homogeneous Neumann problem is precomputed (self.transmission_matrix).
        Dirichlet boundary conditions/average condition are set with Lagrange multipliers.
        """
        if self.boundary_conditions[0] == 'D':
            dirichlet_left = np.zeros(self.nb_face_unknowns)
            dirichlet_left[0] = 1
            dirichlet_left = dirichlet_left[np.newaxis]
            dirichlet_left = sp.coo_array(dirichlet_left)
        if self.boundary_conditions[1] == 'D':
            dirichlet_right = np.zeros(self.nb_face_unknowns)
            dirichlet_right[-1] = 1
            dirichlet_right = dirichlet_right[np.newaxis]
            dirichlet_right = sp.coo_array(dirichlet_right)
        match self.boundary_conditions:
            case 'NN':
                A = sp.block_array([[self.transmission_matrix, self.transmission_average.T], [self.transmission_average, None]], format='csc')
            case 'DN':
                A = sp.block_array([[self.transmission_matrix, dirichlet_left.T], [dirichlet_left, None]], format='csc')
            case 'ND':
                A = sp.block_array([[self.transmission_matrix, dirichlet_right.T], [dirichlet_right, None]], format='csc')
            case 'DD':
                A = sp.block_array([[self.transmission_matrix, dirichlet_left.T, dirichlet_right.T], [dirichlet_left, None, None], [dirichlet_right, None, None]], format='csc')
            case _:
                raise ValueError(f"Unsupported boundary conditions [{self.boundary_conditions}] for global transmission problem.")
        self._transmission_system = sp.linalg.splu(A)

    @property
    def transmission_matrix(self):
        if self._transmission_matrix is None:
            self._build_transmission_matrix()
        return self._transmission_matrix

    def _build_transmission_matrix(self):
        """Build discrete linear system of the global transmission problem with Neumann conditions."""
        # Compute the three nonzero diagonals
        upper_diag = -1/self.spacing 
        lower_diag = upper_diag
        diag = np.zeros(self.nb_face_unknowns)
        diag[:-1] = 1/self.spacing 
        diag[1:] += 1/self.spacing
        # Save the linear system as a sparse matrix
        self._transmission_matrix = sp.diags_array([upper_diag, diag, lower_diag], offsets=[+1, 0, -1])

    @property 
    def transmission_average(self):
        if self._transmission_average is None:
            self._build_transmission_average()
        return self._transmission_average

    def _build_transmission_average(self):
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
        self._transmission_average = sp.coo_array(average)
        
    def _build_transmission_rhs(self, f):
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

    def solve_transmission(self, f, bc_left = None, bc_right = None, average = None):
        """
        Solve the global transmission problem and store the solution in self.solution_face.

        Parameters
        ----------
            f : callable
                Source term of the Poisson equation.
                Should accept an ndarray and return an ndarray of the same size.
            bc_left : double | None, default=None
                Value of the Dirichlet boundary condition on the left.
                Default implements homogeneous condition.
            bc_right : double | None, default=None
                Value of the Dirichlet boundary condition on the right.
                Default implements homogeneous condition.
            average : double | None, default=None
                Average value that is prescribed in the case of Neumann boundary conditions on both sides.
                Default sets average to zero.

        Warns
        --------
            RuntimeWarning
                If the boundary conditions described by bc_left, bc_right and average do not correspond to self.boundary_conditions settings.
        """
        for ibc in range(2):
            bc_check, side = (bc_left, "left") if ibc==0 else (bc_right, "right")
            match self.boundary_conditions[ibc]:
                case 'D':
                    if bc_check is None:
                        if ibc==0:
                            bc_left = 0
                        else:
                            bc_right = 0
                        warnings.warn(f"No Dirichlet condition is provided at the {side}, homogeneous condition is used.", RuntimeWarning)
                case 'N':
                    if bc_check is not None:
                        warnings.warn(f"Neumann condition at the {side} is specified, but only homogeneous conditions are supported.", RuntimeWarning)
        match self.boundary_conditions:
            case 'NN':
                if average is None:
                    raise ValueError("Neumann system requires specification of the average to be solvable.")
            case _:
                if not average is None:
                     raise ValueError(f"Average of the solution cannot be imposed with boundary conditions [{self.boundary_conditions}]")
        # Build rhs of the transmission system for homogeneous Neumann
        self._build_transmission_rhs(f)
        # Add constraints to set boundary conditions/average
        match self.boundary_conditions:
            case 'NN':
                constraints = [average]
            case 'DN':
                constraints = [bc_left]
            case 'ND':
                constraints = [bc_right]
            case 'DD':
                constraints = [bc_left, bc_right]
        rhs = np.concatenate((self.transmission_rhs, np.array(constraints)))
        # Solve
        sol = self.transmission_system.solve(rhs)
        # 
        self.solution_face = sol[0:self.nb_face_unknowns]


    