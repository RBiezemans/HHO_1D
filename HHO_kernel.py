import numpy as np
import scipy.special as sp_fn # special functions, used for Legendre polynomials
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import solve_triangular as solve_tr
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
        nb_cells : int
            Number of cells of the discrete domain.
        cell_degree : int
            Polynomial degree of the cell unknowns.
        basis_type : str
                Description of the basis functions to be used (case-insensitively). Allowed values are:
                - "monomial" corresponds to the monomial basis.
                - "legendre" corresponds to the Legendre basis.
        cells : list of HHO_cell
            Discretization and solution of all the cell problems.
        boundary_conditions : str
            Description of the boundary conditions at both ends of the domain.
            Format 'XX' with X either N (Neumann) or D (Dirichlet), and the left (resp. right) symbol denotes the BC at the left (resp. right).
            Note that for 'NN' the solution is unique up to the average.
        source : callable
            Source term for the Poisson equation.
        transmission_matrix : scipy.sparse.spmatrix
            Matrix associated to the global Neumann problem.
        transmission_average : scipy.sparse.spmatrix
            Row vector representing the average operator in the global transmission problem.
        transmission_average : scipy.sparse.spmatrix
            Matrix for the global problem including boundary conditions through Lagrange multipliers.
        transmission_rhs : ndarray
            Discrete right-hand side for the Neumann system associated to the source term.
        solution_face : ndarray
            Solution vector of the transmission problem.
    """

    def __init__(self, x, degree=0, basis="Monomial"):
        """
        Initialize HHO kernel: define parameters of the grid and the cell degree.

        Args:
            x : ndarray
                vertices of the discrete domain
            degree : int, default = 0
                polynomial degree of the cell unknowns
            basis : str, optional
                Description of the basis functions to be used (case-insensitively). Allowed values are:
                - "monomial" corresponds to the monomial basis.
                - "legendre" corresponds to the Legendre basis.
                Defaults to "Monomial"
        """
        self.points = x
        self.spacing = x[1:] - x[0:-1]

        self.nb_face_unknowns = len(self.points)
        self.nb_cells = self.nb_face_unknowns-1
        self.cell_degree = degree
        self.basis_type = basis

        # Initialize cells problems
        self.cells = [HHO_cell(self.points[i], 
                               self.points[i+1], 
                               self.cell_degree, 
                               self.basis_type) 
                      for i in range(self.nb_face_unknowns-1)]

        # Initialize quantities related to the linear system to unset state
        self._transmission_matrix = None 
        self._transmission_average = None
        self._transmission_system = None
        self._source = None
        self._transmission_rhs = None
        self._boundary_conditions = None
        self._solution_face = None


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
        """Matrix of the discrete linear system of the global transmission problem with Neumann conditions."""
        if self._transmission_matrix is None:
            self._build_transmission_matrix()
        return self._transmission_matrix

    def _build_transmission_matrix(self):
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

    @property 
    def source(self):
        """
        Source term for the Poisson equation.

        Changing the source term ensures the right-hand side of the linear system is recomputed.
        """
        assert self._source is not None, "Right-hand side is required but has not yet been set."
        return self._source
    
    @source.setter
    def source(self, f):
        self._source = f
        self._transmission_rhs = None 

    @property
    def transmission_rhs(self):
        """
        Discrete right-hand side associated to self.source.
        """
        if self._transmission_rhs is None:
            self._build_transmission_rhs(self.source)
        return self._transmission_rhs
        
    def _build_transmission_rhs(self, f):
        """
        Build discrete right-hand side associated to the function f.

        Parameters
        ----------
            f : callable
                Source term of the Poisson equation.
                Should accept an ndarray and return an ndarray of the same size 
        """
        transmission_rhs = np.zeros(self.nb_face_unknowns)
        if self.cell_degree == 0:
            # Compute approximation of the average value of f
            f_half_average = 0.5 * self.spacing * (f(self.points[0:-1]) + f(self.points[1:]))/2.0
            transmission_rhs[0:-1] = f_half_average 
            transmission_rhs[1:] = transmission_rhs[1:] + f_half_average
        else:
            # We do NOT reconstruct the global problem from the local problems as follows
            # for i in range(self.nb_cells):
            #     cell = self.cells[i]
            #     rhs = cell.compute_integral_against_basis(f)
            #     rhs = cho_solve((cell.local_cho, cell._local_cho_lower), rhs, overwrite_b=False)
            #     rhs = cell.HHO_dofs_to_reconstruction[-cell.number_faces:,:cell.degree+1] @ rhs
            #     i_faces = [i, i+1]
            #     transmission_rhs[i_faces] -= rhs
            #
            # We should directly integrate f in the 1D case
            for (i,cell) in enumerate(self.cells):
                quad_degree = int(np.ceil((self.cell_degree)/2)) + 1
                quad_points, quad_weights = cell.quadrature(quad_degree)
                for k in range(cell.number_faces):
                    if k == 0:
                        test = lambda x: 1 - (x - cell.x_left)/cell.h 
                    elif k==1:
                        test = lambda x: 1 + (x - cell.x_right)/cell.h 
                    else:
                        raise RuntimeError("Only 2 faces are supported for 1D HHO.")
                    transmission_rhs[i+k] += np.dot(quad_weights, test(quad_points)*f(quad_points))
        self._transmission_rhs = transmission_rhs
    
    @property 
    def solution_face(self):
        """
        Solution to the global transmission problem.

        Can only be set through the solve_transmission method.
        """
        assert self._solution_face is not None, "Solution to the transmission problem has not yet been computed."
        return self._solution_face

    def solve_transmission(self, f, bc_left = None, bc_right = None, average = None):
        """
        Solve the global transmission problem.
         
        Solution is stored in self.solution_face.

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
        -----
            UserWarning
                If the boundary conditions described by bc_left, bc_right and average do not correspond to self.boundary_conditions settings.
        """
        self.source = f
        for ibc in range(2):
            bc_check, side = (bc_left, "left") if ibc==0 else (bc_right, "right")
            match self.boundary_conditions[ibc]:
                case 'D':
                    if bc_check is None:
                        if ibc==0:
                            bc_left = 0
                        else:
                            bc_right = 0
                        warnings.warn(f"No Dirichlet condition is provided at the {side}, homogeneous condition is used.")
                case 'N':
                    if bc_check is not None:
                        warnings.warn(f"Neumann condition at the {side} is specified, but only homogeneous conditions are supported.")
        match self.boundary_conditions:
            case 'NN':
                if average is None:
                    average = 0
                    warnings.warn("Neumann system requires specification of the average to be solvable. It is set to 0.", RuntimeWarning)
            case _:
                if not average is None:
                    warnings.warn(f"Average of the solution cannot be imposed with boundary conditions [{self.boundary_conditions}]", RuntimeWarning)
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
        rhs = np.concatenate([self.transmission_rhs, np.array(constraints)])
        # Solve
        sol = self.transmission_system.solve(rhs)
        # Save solution to instance
        self._solution_face = sol[0:self.nb_face_unknowns]
    
    def solve_cell_problems(self):
        """
        Solve the local cell problems.

        Requires the prior solution of the global transmission problem.
        """
        for i, cell in enumerate(self.cells):
            cell.solve(self.source, self.solution_face[i], self.solution_face[i+1])

    def solve(self, f, bc_left = None, bc_right = None, average = None):
        """
        Solve the global transmission problem followed by the cell problems.
         
        The global solution is stored in self.solution_face, the local solutions in the self.cells.

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
        """
        self.solve_transmission(f, bc_left, bc_right, average)
        self.solve_cell_problems()


    def plot(self, ax, *args):
        """
        Plot HHO solution/reconstruction on the provided Matplotlib axis.

        Parameters
        ----------
            ax : matplotlib.axes.Axes
                Axis to draw the plot on.
            *args
                Additional arguments to specify which elements to plot. Should be one of:
                * "faces", to plot the solution at the faces;
                * "cells", to plot the solution at the cells;
                * "reconstruction", to plot the reconstruction of the potential that
                  corresponds to the solution at the faces and cells.
                When no additional positional arguments are provided all of the above  are plotted.
        """
        if len(args)==0:
            args = ("faces", "cells", "reconstruction")
        args = [arg.lower() for arg in args]
        # Plot solution at the faces
        if "faces" in args:
            ax.plot(self.points, 
                    self.solution_face, 
                    'r.', 
                    markersize=10, 
                    markeredgewidth=1, 
                    label="HHO \u2014 faces")
        N_plot_cell = 30
        # Plot solution at the cells
        if "cells" in args:
            for i, cell in enumerate(self.cells):
                xx = np.linspace(cell.x_left, cell.x_right, N_plot_cell + cell.degree)
                sol_xx, _ = cell.evaluate_solution(xx, "basis")
                plt_solution, = ax.plot(xx, sol_xx,
                                        color='#1f77b4', 
                                        linewidth=2)
                if i == 0:
                    plt_solution.set_label("HHO \u2014 cells")
        # Plot reconstructed potential in the higher-order space
        if "reconstruction" in args:
            for i, cell in enumerate(self.cells):
                xx = np.linspace(cell.x_left, cell.x_right, N_plot_cell + cell.degree_reconstruction)
                rec_xx, _ = cell.evaluate_reconstruction(xx, "basis")
                plt_reconstruction, = ax.plot(xx, rec_xx,
                                              color='orange', 
                                              linewidth=2)
                if i == 0:
                    plt_reconstruction.set_label("HHO \u2014 reconstruction")
        ax.set_title(f"Approximation by HHO method (degree {self.cell_degree})")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")


class HHO_cell:    
    """
    Handles all information regarding local HHO problems on a single cell.

    Attributes
    ----------
        x_left : double
            Left vertex of the cell.
        x_right : double
            Right vertex of the cell.
        barycenter : double
            Barycenter of the cell.
        h : double
            length of the cell.
        degree : int
            Polynomial degree of the cell unknowns.
        basis_type : str
            Description of the basis functions to be used. Allowed values are:
            - "monomial" corresponds to the monomial basis.
            - "legendre" corresponds to the Legendre basis.
        stabilization_type : int
            Selection of way of stabilization. Allowed values are:
            - 0, to penalize only the difference between cell unknowns and face unknowns at the boundary.
            - 1, to add penalization for the higher-order reconstruction.
        solution : ndarray
            Solution to the local problem on the cell in the polynomial basis.
        solution_faces : ndarray
            Solution at the faces of the cell, ordered from left to right.
        solution_reconstruction
            Reconstruction of the solution to the local problem on the cell in the polynomial basis.
        mass_matrix : ndarray
            Mass matrix of the polynomial basis of degree self.degree on the cell.
        mass_cho : scipy.lingal.cho_factor
            Cholesky factorization of mass_matrix.
        degree_reconstruction : int
            Polynomial degree of the space for the potential reconstruction.
        cell_reconstruction : HHO_cell
            HHO cell on the same domain but with the degree of the reconstruction space.
        stiffness_matrix : ndarray
            Stiffness matrix for the basis of the cell.
        stiffness_matrix_reduced : ndarray
            Stiffness matrix with the constant removed from the basis.
        stiffness_reduced_cho : scipy.linalg.cho_factor
            Cholesky factorization of stiffness_matrix_reduced.
        reconstruction_matrix : ndarray
            Matrix to compute the reconstruction from the HHO cell and face unknowns.
        HHO_dofs_to_reconstruction : ndarray
            Matrix to compute the right-hand side of the reconstruction problem on the cell from the HHO DOFs.
        stabilization_matrix(self):
            Matrix of the stabilization operator.
        local_matrix : ndarray
            Matrix of discrete system for the cell problem including the face unknowns.
        local_cho(self):
            Cholesky decomposition of local_matrix without the face unknowns.
    """

    def __init__(self, x_left, x_right, degree, basis="Monomial", stabilization=1):
        """
        Define parameters for local cell problem.

        Parameters
        ----------
            x_left : double
                Left vertex of the cell.
            x_right : double
                Right vertex of the cell.
            degree : int
                Polynomial degree of the cell unknowns.
            basis : str, optional
                Description of the basis functions to be used (case-insensitively). Allowed values are:
                - "monomial" corresponds to the monomial basis.
                - "legendre" corresponds to the Legendre basis.
                Defaults to "Monomial"
            stabilization : int, optional
                Option to select the desired type of stabilization. Allowed values are:
                - 0, to penalize only the difference between cell unknowns and face unknowns at the boundary.
                - 1, to add penalization for the higher-order reconstruction.
                Defaults to 1.
            
        Raises
        ------
        ValueError  
            If basis is not 
        """
        if not x_left < x_right:
            raise ValueError("Left vertex of cell must be smaller than the right vertex.")
        self.x_left = x_left 
        self.x_right = x_right 
        self.barycenter = (self.x_left + self.x_right)/2.0
        self.h = x_right - self.x_left
        self.degree = degree
        self.degree_reconstruction = self.degree+1
        self._cell_reconstruction = None
            # polynomial degree of the space for the reconstruction of the potential
        self.number_faces = 2

        if basis.lower() not in ["monomial", "legendre"]:
            raise ValueError(f"Unsupported basis type for the cell space ({basis}).")
        self.basis_type = basis.lower()
        if stabilization not in [0,1]:
            raise ValueError(f"Unsupported choice for stabilization ({basis} instead of 0 or 1).")
        self.stabilization_type = stabilization

        self._solution = None
        self._solution_faces = None
        self._solution_reconstruction = None
        
        # Matrices for the discrete cell problem
        self._quad_degree_mass = self.degree+1
        self._mass_matrix = None
        self._mass_cho = None
        self._mass_cho_lower = None 
            # self._mass_cho_lower is a flag indicating whether the factor is in the lower or upper triangle 
            #  returned by scipy.linalg.cho_factor
            # It is not meant to be part of the public interface of the class
        self._stiffness_matrix = None
        self._stiffness_matrix_reduced = None
        self._stiffness_reduced_cho = None
        self._stiffness_reduced_cho_lower = None # similar use as self._mass_cho_lower
            # on the faces we would need a higher degree for the quadrature,
            # but in 1D integrals over the faces are simply point-wise evaluation
        self._HHO_dofs_to_reconstruction = None
        self._reconstruction_matrix = None
        self._stabilization_matrix = None
        self._local_matrix = None
        self._local_cho = None 
        self._local_cho_lower = None # similar use as self._mass_cho_lower

    @property 
    def solution(self):
        """
        Solution to the local problem on the cell in the polynomial basis.

        Can only be set through HHO_cell.solve().
        """
        assert self._solution is not None, "Solution to local problem has not yet been computed."
        return self._solution
    
    @property 
    def solution_faces(self):
        """
        Solution at the faces of the cell.
        
        solution_faces[0] is the left face, solution_faces[1] is the right face.
        Can only be set through HHO_cell.solve()
        """
        assert self._solution_faces is not None, "Solution at the cell faces is unknown."
        return self._solution_faces

    @property 
    def solution_reconstruction(self):
        """
        Reconstruction of the solution to the local problem on the cell in the polynomial basis.

        Can only be set through HHO_cell.solve().
        """
        assert self._solution_reconstruction is not None, "Solution to local problem has not yet been computed."
        return self._solution_reconstruction

    def solve(self, source, sol_global_left, sol_global_right):
        """
        Solve local problem based on the source term and the solution to the global solution on both faces of the cell.

        Solution is saved to self.solution.

        Parameters
        ----------
            source : callable
                Source term of the Poisson equation.
                Should accept an ndarray and return an ndarray of the same size.
            sol_global_left : double
                Solution to the global problem at the left vertex of the cell.
            sol_global_right : double
                Solution to the global problem at the right vertex of the cell.
        """
        solution_faces = np.array([sol_global_left, sol_global_right])
        if self.degree == 0:
            # 1st order approximation of the average of the source
            source_average = self.h * source(self.barycenter)
            # 2nd order approximation of the average of the source
            # source_average = self.h * (source(self.x_left) + source(self.x_right))/2.0
            solution = 0.5 * self.h**2 * source_average + 0.5*(sol_global_left + sol_global_right)
            solution = np.reshape(solution,(1))
            # Rescale by the value of the constant basis function in case it might not be 1
            value_basis, _ = self.evaluate_basis(self.barycenter, "basis")
            solution = solution/value_basis[0,0]
        else:
            # Contribution of the source term of the PDE to the local problem
            rhs_volume = self.compute_integral_against_basis(source)
            # Contribution of the face unknowns to the local problem
            rhs_face = self.local_matrix[:,-self.number_faces:] @ solution_faces
            rhs_face = rhs_face[0:-self.number_faces]
            # Build total RHS and solve local problem
            rhs = rhs_volume - rhs_face
            solution = cho_solve((self.local_cho, self._local_cho_lower), rhs, overwrite_b=True)
        self._solution = solution
        self._solution_faces = solution_faces
        self._solution_reconstruction = self.compute_reconstruction(np.concatenate([self.solution, self.solution_faces]))
        
    def quadrature(self, deg):
        """
        Compute the sample points and weights for Gauss-Legendre quadrature on the cell.

        The integration is exact for polynomials up to degree 2*deg-1.

        Parameters
        ----------
            deg : int
                Number of sample points and weights. It must be >= 1.

        Returns
        -------
            x : ndarray
                1-D ndarray containing the sample points.
            y : ndarray
                1-D ndarray containing the weights.
        """
        x_left_reference = -1.0 # length end point of the reference cell (-1,1)
        h_reference = 2.0 # length of the reference cell
        x, w = np.polynomial.legendre.leggauss(deg)
        x = (x - x_left_reference) / h_reference * self.h + self.x_left
        w = w * self.h/h_reference
        return x, w


    def evaluate_basis(self, points, *args):
        """
        Evaluate the basis functions on the cell and their derivatives at prescribed points.

        Parameters
        ----------
            points : ndarray | scalar type
                The points in which the basis is to be evaluated.
            *args
                Strings indicating which quantities to compute: "basis"/"gradient".
                When nothing is specified, compute both.
        
        Returns
        -------
        ndarray
            Values of the basis functions if "basis" is passed in *args, otherwise None.
            The array is of the shape (N_points, N_basis): the first axis corresponds to 
            different evaluation points and the second axis to different basis functions.
            The constant basis function (kernel of the gradient) is the first basis function.

        ndarray
            Values of the gradient if "gradient" is passed in *args, otherwise None.
            Follows the format of the first output.
        """
        basis_value = None
        gradient_value = None 
        #
        degree = self.degree
        compute_all = len(args) == 0
        compute_basis = "basis" in args or compute_all
        compute_gradient = "gradient" in args or compute_all
        if np.isscalar(points) or points.ndim == 0:
            points = np.reshape(points,(1,1))

        if degree == 0:
            if compute_basis:
                basis_value = np.ones((len(points),1))
            if compute_gradient:
                gradient_value = np.zeros((len(points),1))
        else:
            N_points = len(points)
            # Rescale evaluation points to the refernce cell (-1,1)
            h_reference = 2.0
            points_scaled = (points-self.barycenter)*h_reference/self.h
            #
            match self.basis_type:
                case "monomial":
                    powers = np.arange(0,degree+1)
                    # Reshape points and powers to take advantage of numpy broadcasting
                    points_scaled = points_scaled[:, None]
                    powers = powers[None, :]
                    # Compute value of the basis functions
                    if compute_basis:
                        basis_value = points_scaled ** powers
                    # Compute value of the derivatives
                    gradient_value = np.zeros((N_points,degree+1)) 
                    # The derivative of the first basis function is zero everywhere
                    if compute_gradient:
                        gradient_value[:,1:] = (points_scaled[:] ** powers[:,:-1]) * (powers[0,1:] * h_reference/self.h)
                case "legendre":
                    nb_basis_functions = degree+1
                    if compute_basis:
                        basis_value = np.zeros((N_points,nb_basis_functions))
                        for i in range(nb_basis_functions):
                            basis_value[:,i] = sp_fn.eval_legendre(i, points_scaled)
                    if compute_gradient:
                        gradient_value = np.zeros((N_points,nb_basis_functions))
                        for i in range(nb_basis_functions):
                            gradient = sp_fn.legendre(i).deriv()
                            gradient_value[:,i] = gradient(points_scaled)*h_reference/self.h
        return basis_value, gradient_value
    
    def evaluate_fun(self, points, f, *args):
        """
        Evaluate a function given by its coefficients in the basis at prescribed points and its gradient.

        Parameters
        ----------
            points : ndarray
                The points in which the basis is to be evaluated.
            f : ndarray
                List of coefficients of the function to be evaluated with respect to the basis.
            *args
                Strings indicating which quantities to compute: "basis"/"gradient".

        Returns
        -------
        ndarray
            Values of the function at the given points if "basis" is passed in *args, otherwise None.
        ndarray
            Values of the gradient of the function at the given points if "gradient" is passed in *args, otherwise None.
        """
        basis, gradient = self.evaluate_basis(points, *args)
        if basis is not None:
            basis = basis @ f
        if gradient is not None:
            gradient = gradient @ f
        return basis, gradient

    def evaluate_solution(self, points, *args):
        """
        Evaluate the cell unknowns of the HHO solution and the corresponding gradient.

        Parameters
        ----------
            points : ndarray
                The points in which the basis is to be evaluated.
            *args
                Strings indicating which quantities to compute: "basis"/"gradient".

        Returns
        -------
        ndarray
            Values of the solution at the given points if "basis" is passed in *args, otherwise None.
        ndarray
            Values of the gradient of the solution at the given pointsif "gradient" is passed in *args, otherwise None.
        """
        return self.evaluate_fun(points, self.solution, *args)

    def evaluate_reconstruction(self, points, *args):
        """
        Evaluate the reconstruction corresponding to the HHO solution and its gradient in the higher-order space.

        Parameters
        ----------
            points : ndarray
                The points in which the basis is to be evaluated.
            *args
                Strings indicating which quantities to compute: "basis"/"gradient".

        Returns
        -------
        ndarray
            Values of the reconstruction at the given points if "basis" is passed in *args, otherwise None.
        ndarray
            Values of the gradient of the reconstruction at the given points if "basis" is passed in *args, otherwise None.
        """
        return self.cell_reconstruction.evaluate_fun(points, self.solution_reconstruction, *args)
    
    @property 
    def mass_matrix(self):
        """Mass matrix of the polynomial basis of degree self.degree on the cell."""
        if self._mass_matrix is None:
            self._build_mass_matrix()
        return self._mass_matrix
    
    def _build_mass_matrix(self):
        # Evaluate basis functions at the quadrature points
        quad_points, quad_weights = self.quadrature(self._quad_degree_mass)
        basis, _ = self.evaluate_basis(quad_points, "basis")
        # Prepare basis-basis multiplication at each quadrature point
        basis_column = basis[:,:,None]
        basis_row = basis[:,None,:]
        # Compute basis-basis multiplication
        mass_matrix = np.einsum('ijk,ikl->ijl', basis_column, basis_row)
        # Integrate mass matrix
        self._mass_matrix = np.einsum('i,ikl->kl', quad_weights, mass_matrix)
        # Reset the Cholesky decomposition of the mass matrix
        self._mass_cho = None
        self._mass_cho_lower = None

    @property 
    def mass_cho(self):
        """Cholesky factorization of the mass matrix of the polynomial basis of degree self.degree on the cell."""
        if self._mass_cho is None:
            self._build_mass_cho()
        return self._mass_cho
    
    def _build_mass_cho(self):
        self._mass_cho, self._mass_cho_lower = cho_factor(self.mass_matrix, overwrite_a=True)

    def compute_L2_projection(self, f):
        """
        Compute the L2-orthogonal projection of f on the set of basis polynomials of the cell.

        Parameters
        ----------
            f : callable
                Function that we want to compute the interpolate of.
                Should accept an ndarray and return an ndarray of the same size.
        
        Returns
        -------
            ndarray
                L2-orthogonal projection of f in the form of its coefficients in the basis.
        """
        # Compute RHS for linear system of the L2 projection
        rhs = self.compute_integral_against_basis(f)
        # Solve projection problem
        return self.compute_L2_projection_from_rhs(rhs)

    def compute_L2_projection_from_rhs(self, rhs):
        """
        Compute the L2-orthogonal projection on the cell given the right-hand side vector of the system.

        Parameters
        ----------
            rhs : ndarray
                Right-hand side of the linear system of the L2-projecetion
        
        Returns
        -------
            ndarray
                L2-orthogonal projection of f in the form of its coefficients in the basis.
        """
        return cho_solve((self.mass_cho, self._mass_cho_lower), rhs)

    def compute_integral_against_basis(self, f):
        """
        Compute the integral of a function against all basis functions of the cell.

        The output is the source term of e.g. the L2-projection and the reconstruction problem.

        Parameters
        ----------
            f : callable
                Function to be integrated
                Should accept an ndarray and return an ndarray of the same size.

        Returns
        -------
            ndarray
                Integral against each basis function.
        """
        quad_points, quad_weights = self.quadrature(self._quad_degree_mass)
        f_eval = f(quad_points)
        basis, _ = self.evaluate_basis(quad_points, "basis")
        fbasis = basis * f_eval[:,None] # Multiply each basis function by f
        # Compute integrals
        return np.einsum('i,ik->k', quad_weights, fbasis)

    @property
    def stiffness_matrix(self):
        """Stiffness matrix on the basis of the cell."""
        if self._stiffness_matrix is None:
            self._build_stiffness_matrix()
        return self._stiffness_matrix
    
    def _build_stiffness_matrix(self):
        # Evaluate gradient of the basis functions at the quadrature points
        quad_points, quad_weights = self.quadrature(self.degree)
        _, dbasis = self.evaluate_basis(quad_points, "gradient")
        # Prepare basis-basis multiplication at each quadrature point
        dbasis_column = dbasis[:,:,None]
        dbasis_row = dbasis[:,None,:]
        # Compute basis-basis multiplication
        stiffness = np.einsum('ijk,ikl->ijl', dbasis_column, dbasis_row)
        # Integrate stiffness matrix
        self._stiffness_matrix = np.einsum('i,ikl->kl', quad_weights, stiffness)
        # Reset the dependencies of the stiffness matrix
        self._stiffness_matrix_reduced = None
        self._stiffness_reduced_cho = None
        self._stiffness_reduced_cho_lower = None

    @property 
    def stiffness_matrix_reduced(self):
        """Stiffness matrix with the constant function removed from the basis."""
        return self.stiffness_matrix[1:,1:]

    @property 
    def stiffness_reduced_cho(self):
        """Cholesky factorization of self.stiffness_matrix_reduced."""
        if self._stiffness_reduced_cho is None:
            self._stiffness_reduced_cho, self._stiffness_reduced_cho_lower = cho_factor(self.stiffness_matrix_reduced)
        return self._stiffness_reduced_cho

    @property 
    def HHO_dofs_to_reconstruction(self):
        """
        Matrix to compute the right-hand side of the reconstruction problem on the cell from the HHO DOFs.

        The DOFs = of the HHO method on the cell are assumed to be ordered as follows: 
            - L2 projections on the cell 
            - L2 projections on the left face (i.e., the value in 1D)
            - L2 projections on the right face (i.e., the value in 1D)
        """
        if self._HHO_dofs_to_reconstruction is None:
            self._build_HHO_dofs_to_reconstruction()
        return self._HHO_dofs_to_reconstruction
    
    def _build_HHO_dofs_to_reconstruction(self):
        # Initialization 
        source = np.zeros((self.degree+1, self.degree+1+self.number_faces))
        ###
        ### Integrals over the cell volume
        ###
        ### Compute (dbasis,dbasis_rec)_L2(cell), trial functions in the cell space
        # Evaluate gradient of the basis functions at the quadrature points
        # We need both the basis functions from the HHO space and the higher-order reconstruction space.
        quad_degree = self.degree
        quad_points, quad_weights = self.quadrature(quad_degree)
        basis_all, dbasis_all = self.cell_reconstruction.evaluate_basis(quad_points)
        basis = basis_all[:,:self.degree+1]
        dbasis = dbasis_all[:,:self.degree+1]
        dbasis_rec = dbasis_all[:,1:] # Drop constant function from the basis of the reconstruction space
        # Compute product of bases
        dbasis_row = dbasis[:,None,:]
        dbasis_rec_column = dbasis_rec[:,:,None]
        cont = np.einsum('ijk,ikl->ijl', dbasis_rec_column, dbasis_row)
        # Apply quadrature
        cont = np.einsum('i,ikl->kl', quad_weights, cont)
        ### Store contribution to source
        source[:,:self.degree+1] = cont
        ###
        ### Integrals over the faces
        ###
        ### Compute (basis,dbasis)_L2(faces), trial functions in the cell space
        # Evaluate basis functions at the faces
        faces = np.array((self.x_left, self.x_right))
        faces_normal = np.array([-1,1])
        basis, _ = self.evaluate_basis(faces, "basis")
        _, dbasis_rec = self.cell_reconstruction.evaluate_basis(faces, "gradient")
        dbasis_rec = dbasis_rec[:,1:]
        # Compute product of bases
        basis_row = basis[:,None,:]
        dbasis_rec_column = dbasis_rec[:,:,None]
        cont = np.einsum('ijk,ikl->ijl', dbasis_rec_column, basis_row)
        ### Store contributions to source
        for k in range(len(faces)):
            source[:,:self.degree+1] -= faces_normal[k] * cont[k,:,:]
        ### Compute (basis,dbasis_rec)_L2(faces), trial function in the face space
        for k in range(len(faces)):
            source[:,self.degree+1+k] = faces_normal[k] * dbasis_rec_column[k,:,0]
        ### Done
        self._HHO_dofs_to_reconstruction = source
    
    @property 
    def reconstruction_matrix(self):
        """Matrix to compute the reconstruction from the HHO cell and face unknowns."""
        if self._reconstruction_matrix is None:
            self._reconstruction_matrix = cho_solve((self.cell_reconstruction.stiffness_reduced_cho, self.cell_reconstruction._stiffness_reduced_cho_lower), 
                                                     self.HHO_dofs_to_reconstruction)
        return self._reconstruction_matrix        

    @property 
    def cell_reconstruction(self):
        """HHO cell on the same domain but with the degree of the reconstruction space. """
        if self._cell_reconstruction is None:
            if self.degree == self.degree_reconstruction:
                self._cell_reconstruction = self
            else:
                self._cell_reconstruction = HHO_cell(self.x_left, self.x_right, self.degree_reconstruction, self.basis_type)
        return self._cell_reconstruction

    def compute_reconstruction(self, dofs):
        """
        Compute higher-order reconstruction in the polynomial basis corresponding to given HHO cell and face DOFs.

        Parameters
        ----------
            dofs : ndarray
                Degrees of freedom of the HHO method on the cell, consisting of the L2 projection
                on the cell basis followed by the value at the left face, then the right face.

        Returns
        -------
            Higher-order reconstruction based on the provided DOS in the form of its coefficients in the basis.
        """
        if self.degree == 0:
            # We use the explicit solution of the reconstruction problem available in this case
            gradient = (dofs[-1]-dofs[-2])/self.h
            r = lambda x: dofs[0] + gradient*(x-self.barycenter)
            # We need to manually project the reconstruction on the reconstruction space
            return self.cell_reconstruction.compute_L2_projection(r)
        else:
            # solve_reconstruction
            reconstruction = self.reconstruction_matrix @ dofs
            # Compute current integral of the reconstruction (without the constant function)
            quad_average_x, quad_average_w = self.quadrature(self.degree_reconstruction+1)
            basis_r, _ = self.cell_reconstruction.evaluate_basis(quad_average_x, "basis")
            integral_r = np.dot(basis_r[:,1:] @ reconstruction, quad_average_w)
            # Compute integral of the constant basis function
            integral0 = np.dot(basis_r[:,0], quad_average_w)
            # Compute integral of the function described by the DOFs
            integral_dof = np.dot(basis_r[:,:-1] @ dofs[:-self.number_faces], quad_average_w)
            # Compute coefficient in front of the constant basis function
            c0 = np.array((integral_dof - integral_r)/integral0, ndmin=1)
            return np.concatenate([c0, reconstruction])
    
    @property 
    def stabilization_matrix(self):
        """Matrix of the stabilization operator."""
        if self._stabilization_matrix is None:
            self._build_stabilization_matrix()
        return self._stabilization_matrix

    def _build_stabilization_matrix(self):
        ###
        ### Penalty on the jump between the cell and trace unknowns
        ###
        stabilization = np.zeros((self.number_faces, self.degree+1+self.number_faces))
        face_coordinates = np.array([self.x_left, self.x_right])
        basis_rec, _ = self.cell_reconstruction.evaluate_basis(face_coordinates, "basis")
        basis = basis_rec[:,:self.degree+1]
        # Subtract DOF at the face
        stabilization[:,self.degree+1:] = -1*np.eye(self.number_faces)
        # Add volume contribution at the face
        stabilization[:,:self.degree+1] = basis
        ###
        ### Penalty on the high-order reconstruction
        ###
        if self.stabilization_type == 1:
            # Note that we can neglect the contribution to the stabilization of the
            # constant basis function in the reconstruction for self.degree >= 1
            #
            # Drop constant basis function from the reconstruction space
            basis_rec = basis_rec[:,1:]
            # Compute contribution to stabilization matrix
            S_rec = basis_rec @ self.reconstruction_matrix
            stabilization += S_rec
            # Compute transfer matrix from reconstruction to projection on the cell
            quad_points, quad_weights = self.quadrature(self._quad_degree_mass)
            basis_rec_on_cell, _ = self.cell_reconstruction.evaluate_basis(quad_points, "basis")
            basis_on_cell = basis_rec_on_cell[:,:self.degree+1]
            # Drop constant basis function from reconstruction space basis
            basis_rec_on_cell = basis_rec_on_cell[:,1:]
            # Prepare bases vectors for multiplication
            basis_on_cell = basis_on_cell[:,:,None] 
            basis_rec_on_cell = basis_rec_on_cell[:,None,:]
            transfer = np.einsum('ijk,ikl->ijl', basis_on_cell, basis_rec_on_cell)
            # Compute transfer matrix by integration
            transfer = np.einsum('i,ijk->jk', quad_weights, transfer)
            # Compute contribution to stabilization matrix
            S_proj_rec = transfer @ self.reconstruction_matrix
            S_proj_rec = self.compute_L2_projection_from_rhs(S_proj_rec)
            S_proj_rec = basis @ S_proj_rec
            stabilization -= S_proj_rec
        self._stabilization_matrix = stabilization

    @property 
    def local_matrix(self):
        """Matrix of discrete system for the local cell problem including the face unknowns."""
        if self._local_matrix is None:
            self._build_local_matrix()
        return self._local_matrix
    
    def _build_local_matrix(self):
        H = self.HHO_dofs_to_reconstruction
        local_matrix = H.T @ self.reconstruction_matrix
        # Stabilization contributions
        for k in range(self.number_faces):
            S = self.stabilization_matrix[k,:]
            S = S[:,None] @ S[None,:] 
            local_matrix += 1.0/self.h * S
        self._local_matrix = local_matrix
        self._local_cho = None
        self._local_cho_lower = None

    @property 
    def local_cho(self):
        """Cholesky decomposition of local_matrix."""
        if self._local_cho is None:
            self._compute_local_cho()
        return self._local_cho 
    
    def _compute_local_cho(self):
        # The local problem uses homogeneous values for the face unknowns
        local_matrix = self.local_matrix[:-self.number_faces,:-self.number_faces]
        self._local_cho, self._local_cho_lower = cho_factor(local_matrix, overwrite_a=True)