import numpy as np
import scipy.special as sp_fn # special functions, used for Legendre polynomials
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
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

    Methods
    -------
        solve_transmission(f, bc_left=None, bc_right=None, average=None)
            Solve the global transmission problem with source term f.
            Non-homogeneous Dirichlet boundary conditions are set by specifying bc_left, bc_right.
            In case of pure Neumann conditions, the average of the solution is set by specifying average.
        solve_cell_problems()
            Solve the cell problem on all cells.
            Requires the prior solution of the transmission problem.
        solve(f, bc_left=None, bc_right=None, average=None)
            Solve the transmission problems followed by the cell problems.
            The same comments as for solve_transmission apply
        plot(ax)
            Plot solution at the faces and on the cells on the provided Matplotlib axis.
    """

    def __init__(self, x, degree=0):
        """
        Initialize HHO kernel: define parameters of the grid and the cell degree.

        Args:
            x : ndarray
                vertices of the discrete domain
            degree : int
                polynomial degree of the cell unknowns
        """
        self.points = x
        self.spacing = x[1:] - x[0:-1]

        self.nb_face_unknowns = len(self.points)
        self.nb_cells = self.nb_face_unknowns-1
        self.cell_degree = degree

        # Initialize cells problems
        self.cells = [HHO_cell(self.points[i], self.points[i+1], self.cell_degree) 
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

    @property 
    def source(self):
        """
        Source term for the Poisson equation.

        Changing the source term ensures the right-hand side of the linear system is recomputed.
        """
        assert self._source is not None, "Right-hand side is requiested but has not yet been set."
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
        if self.cell_degree == 0:
            # Compute approximation of the average value of f
            transmission_rhs = np.zeros(self.nb_face_unknowns)
            f_half_average = 0.5 * self.spacing * (f(self.points[0:-1]) + f(self.points[1:]))/2.0
            transmission_rhs[0:-1] = f_half_average 
            transmission_rhs[1:] = transmission_rhs[1:] + f_half_average
        else:
            raise ValueError("Source term can only be discretized for cell degree 0.")
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
            RuntimeWarning
                If the boundary conditions described by bc_left, bc_right and average do not correspond to self.boundary_conditions settings.
        """
        self.source = f
        self._solution_face = None
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
        rhs = np.concatenate((self.transmission_rhs, np.array(constraints)))
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


    def plot(self, ax):
        """
        Plot HHO solution/reconstruction on the provided Matplotlib axis.

        Parameters
        ----------
            ax : matplotlib.axes.Axes
                Axis to draw the plot on.
        """
        ax.plot(self.points, 
                self.solution_face, 
                'r.', 
                markersize=10, 
                markeredgewidth=1, 
                label="HHO \u2014 faces")
        for i, cell in enumerate(self.cells):
            plt_solution, = ax.plot(cell.points_plot, 
                                    cell.solution_plot, 
                                    color='#1f77b4', 
                                    linewidth=2)
            plt_reconstruction, = ax.plot(cell.points_plot,
                                          cell.reconstruction_plot,
                                          color='orange', 
                                          linewidth=2)
            if i == 0:
                plt_solution.set_label("HHO \u2014 cells")
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
        solution : ndarray
            Solution to the local problem on the cell in the polynomial basis and on the faces.
        solution_faces : ndarray
            Solution at the faces of the cell.
            solution_faces[0] is on the left face, solution_faces[1] on the right face.
        plot_margin : ndarray
            Margin around the faces that is used for visualization of the discontinuous solution.
        points_plot : ndarray
            Points in the cell used to plot the cell solution.
        solution_plot : ndarray
            Representation of the solution in the cell on points_plot.
        reconstruction_plot : ndarray
            Representation of the potential reconstruction in the cell on points_plot.
        mass : ndarray
            Mass matrix of the polynomial basis of degree self.degree on the cell.
        mass_cho : scipy.lingal.cho_factor
            Cholesky factorization of mass.
        degree_reconstruction : int
            Polynomial degree of the space for the potential reconstruction.
        reconstruction_matrix : ndarray
            Stiffness matrix in the space for the potential reconstruction, without the constant function.
        reconstruction_cho : scipy.linalg.cho_factor
            Cholesky factorization of reconstruction_matrix.
        reconstruction_source : ndarray
            Matrix to compute the right-hand side of the reconstruction problem on the cell from the HHO DOFs.

    Methods
    -------
        solve(source, sol_global_left, sol_global_right))
            Solve local problem based on the source term and the solution to the global solution on both faces of the cell.
        quadrature(self, deg):
            Compute the sample points and weights for Gauss-Legendre quadrature on the cell.
        evaluate_basis(points, degree=None)
            Evaluate the basis functions on the cell and their derivatives at prescribed points.
        evaluate_fun(points, f, degree=None)
            Evaluate a function given by its coefficients with respect to the basis at prescribed points.
        compute_L2_projection(f)
            Compute the L2-orthogonal projection of f on the set of basis polynomials of the cell.
        compute_reconstruction(dofs):
            Compute higher-order reconstruction in the polynomial basis corresponding to given HHO cell and face DOFs.
    """

    def __init__(self, x_left, x_right, degree):
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
        """
        if not x_left < x_right:
            raise ValueError("Left vertex of cell must be smaller than the right vertex.")
        self.x_left = x_left 
        self.x_right = x_right 
        self.barycenter = (self.x_left + self.x_right)/2.0
        self.h = x_right - self.x_left
        self.degree = degree
        self.degree_reconstruction = self.degree+1
            # polynomial degree of the space for the reconstruction of the potential

        self._solution = None
        self._solution_faces = None

        self.plot_margin = self.h * 1E-5
        self._points_plot = None
        self._solution_plot = None
        self._reconstruction_plot = None

        # Matrices for the discrete cell problem
        self._quad_degree_mass = self.degree+1
        self._mass = None
        self._mass_cho = None
        self._mass_cho_lower = None 
            # self._mass_cho_lower is a flag indicating whether the factor is in the lower or upper triangle 
            #  returned by scipy.linalg.cho_factor
            # It is not meant to be part of the public interface of the class
        self._quad_degree_reconstruction = self.degree+1
        self._reconstruction_matrix = None
        self._reconstruction_cho = None
        self._reconstruction_cho_lower = None
            # similar use as to self._mass_cho_lower
        self._quad_degree_reconstruction_source = self.degree
            # on the faces we would need a higher degree for the quadrature,
            # but in 1D integrals over the faces are simply point-wise evaluation
        self._reconstruction_source = None

    @property 
    def solution(self):
        """
        Solution to the local problem on the cell in the polynomial basis and on the faces.

        Can only be set through the solve method.
        """
        assert self._solution is not None, "Solution to local problem has not yet been computed."
        return self._solution
    
    @property 
    def solution_faces(self):
        """
        Solution at the faces of the cell.
        
        solution_faces[0] is the left face, solution_faces[1] is the right face.
        Can only be set by self.solve()
        """
        assert self._solution_faces is not None, "Solution at the cell faces is unknown."
        return self._solution_faces

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
        self._solution = np.zeros(self.degree+3)
        if self.degree == 0:
            source_average = self.h * (source(self.x_left) + source(self.x_right))/2.0
            solution = 0.5 * self.h**2 * source_average + 0.5*(sol_global_left + sol_global_right)
        else:
            raise ValueError("Local problems have only been implemented for cell degree 0.")
        self._solution = solution
        self._solution_faces = np.array([sol_global_left, sol_global_right])
        self._solution_plot = None
        self._reconstruction_plot = None

    @property
    def points_plot(self):
        """Points in the cell used to plot the cell solution."""
        if self._points_plot is None:
            self._build_points_plot()
        return self._points_plot
    
    def _build_points_plot(self):
        if self.degree == 0:
            points_plot = np.array([self.x_left+self.plot_margin, self.x_right-self.plot_margin])
        else:
            raise ValueError("Plot of cell solutions has only been implemented for cell degree 0.")
        self._points_plot = points_plot

    @property 
    def solution_plot(self):
        """Representation of the solution in the cell on points_plot."""
        if self._solution_plot is None:
            self._build_solution_plot()
        return self._solution_plot
    
    def _build_solution_plot(self):
        if self.degree == 0:
            solution_plot = self.solution * np.ones(len(self.points_plot))
        else:
            raise ValueError("Plot of cell solutions has only been implemented for cell degree 0.")
        self._solution_plot = solution_plot
    
    @property
    def reconstruction_plot(self):
        """Representation of the potential reconstruction in the cell on points_plot."""
        if self._reconstruction_plot is None:
            self._build_reconstruction_plot()
        return self._reconstruction_plot
    
    def _build_reconstruction_plot(self):
        if self.degree == 0:
            self._reconstruction_plot = self.solution + (self.solution_faces[1]-self.solution_faces[0])*(self.points_plot-self.barycenter)/self.h
        else:
            raise ValueError("Local problems have only been implemented for cell degree 0.")
    
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


    def evaluate_basis(self, points, degree=None):
        """
        Evaluate the basis functions on the cell and their derivatives at prescribed points.

        Parameters
        ----------
            points : ndarray
                The points in which the basis is to be evaluated.
            degree : int | None, default = None
                Polynomial degree of the basis to be evaluated.
                When None, the degree is the value in self.degree.
        
        Returns
        -------
        ndarray
            Values of the basis functions.
            The array is of the shape (N_points, N_basis): the first axis corresponds to 
            different evaluation points and the second axis to different basis functions.
            The constant basis function (kernel of the gradient) is the first basis function.

        ndarray
            Values of the gradient, in the same format as the first output.
        """
        if degree is None:
            degree = self.degree
        N_points = len(points)
        # Rescale evaluation points to the refernce cell (-1,1)
        h_reference = 2.0
        points_scaled = (points-self.barycenter)*h_reference/self.h
        #
        # Monomial basis
        powers = np.arange(0,degree+1)
        # Reshape points and powers to take advantage of numpy broadcasting
        points_scaled = points_scaled[:, None]
        powers = powers[None, :]
        # Compute value of the basis functions
        basis_value = points_scaled ** powers
        # Compute value of the derivatives
        gradient_value = np.zeros((N_points,degree+1)) 
        # The derivative of the first basis function is zero everywhere
        gradient_value[:,1:] = (points_scaled[:] ** powers[:,:-1]) * (powers[0,1:] * h_reference/self.h)
        #
        # Legendre 
        # basis_value = np.zeros((N_points,degree+1))
        # gradient_value = np.zeros(basis_value.shape)
        # for i in range(degree+1):
        #     basis_value[:,i] = sp_fn.eval_legendre(i, points_scaled)
        #     gradient = sp_fn.legendre(i).deriv()
        #     gradient_value[:,i] = gradient(points_scaled)
        #
        return basis_value, gradient_value
    
    def evaluate_fun(self, points, f, degree = None):
        """
        Evaluate a function given by its coefficients in the basis at prescribed points.

        Parameters
        ----------
            points : ndarray
                The points in which the basis is to be evaluated.
            f : ndarray
                List of coefficients of the function to be evaluated with respect to the basis.
            degree : int | None, default = None
                Polynomial degree of the basis to be evaluated.
                When None, the degree used is the value in self.degree.

        Returns
        -------
        ndarray
            Values of the function at the given points.
        """
        basis, _ = self.evaluate_basis(points, degree)
        return basis @ f
    
    @property 
    def mass(self):
        """Mass matrix of the polynomial basis of degree self.degree on the cell."""
        if self._mass is None:
            self._build_mass() # must set self._mass_cho to None
        return self._mass
    
    def _build_mass(self):
        # Evaluate basis functions at the quadrature points
        quad_points, quad_weights = self.quadrature(self._quad_degree_mass)
        basis, _ = self.evaluate_basis(quad_points)
        # Prepare basis-basis multiplication at each quadrature point
        basis_column = basis[:,:,None]
        basis_row = basis[:,None,:]
        # Compute basis-basis multiplication
        mass = np.einsum('ijk,ikl->ijl', basis_column, basis_row)
        # Integrate mass matrix
        self._mass = np.einsum('i,ikl->kl', quad_weights, mass)
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
        self._mass_cho, self._mass_cho_lower = cho_factor(self.mass, overwrite_a=True)

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
        # Compute RHS for the projection linear system
        quad_points, quad_weights = self.quadrature(self._quad_degree_mass)
        f_eval = f(quad_points)
        basis, _ = self.evaluate_basis(quad_points)
        rhs = basis * f_eval[:,None] # Multiply each basis function by f
        rhs = np.einsum('i,ik->k', quad_weights, rhs)
        # Solve projection problem
        return cho_solve((self.mass_cho, self._mass_cho_lower), rhs, overwrite_b=True)

    @property
    def reconstruction_matrix(self):
        """
        Stiffness matrix in the space for the potential reconstruction, without the constant function.
        """
        if self._reconstruction_matrix is None:
            self._build_reconstruction_matrix()
        return self._reconstruction_matrix
    
    def _build_reconstruction_matrix(self):
        # Evaluate gradient of the basis functions at the quadrature points
        quad_points, quad_weights = self.quadrature(self._quad_degree_reconstruction)
        _, dbasis = self.evaluate_basis(quad_points, self.degree_reconstruction)
        # Drop gradient of constant basis function
        dbasis = dbasis[:,1:]
        # Prepare basis-basis multiplication at each quadrature point
        dbasis_column = dbasis[:,:,None]
        dbasis_row = dbasis[:,None,:]
        # Compute basis-basis multiplication
        stiffness = np.einsum('ijk,ikl->ijl', dbasis_column, dbasis_row)
        # Integrate stiffness matrix
        self._reconstruction_matrix = np.einsum('i,ikl->kl', quad_weights, stiffness)
        # Reset the Cholesky decomposition of the stiffness matrix
        self._reconstruction_cho = None
        self._reconstruction_cho_lower = None

    @property
    def reconstruction_cho(self):
        """Cholesky factorization of the matrix in the space for the potential reconstruction (without the constant function)."""
        if self._reconstruction_cho is None:
            self._build_reconstruction_cho()
        return self._reconstruction_cho
    
    def _build_reconstruction_cho(self):
        self._reconstruction_cho, self._reconstruction_cho_lower = cho_factor(self.reconstruction_matrix, overwrite_a=True)
        
    @property 
    def reconstruction_source(self):
        """
        Matrix to compute the right-hand side of the reconstruction problem on the cell from the HHO DOFs.

        The DOFs = of the HHO method on the cell are assumed to be ordered as follows: 
            - L2 projections on the cell 
            - L2 projections on the left face (i.e., the value in 1D)
            - L2 projections on the right face (i.e., the value in 1D)
        """
        if self._reconstruction_source is None:
            self._build_reconstruction_source()
        return self._reconstruction_source
    
    def _build_reconstruction_source(self):
        # Initialization 
        nb_faces = 2
        source = np.zeros([self.degree+1, self.degree+1+nb_faces])
        ###
        ### Integrals over the cell volume
        ###
        ### Compute (dbasis,dbasis_rec)_L2(cell), trial functions in the cell space
        # Evaluate gradient of the basis functions at the quadrature points
        # We need both the basis functions from the HHO space and the higher-order reconstruction space.
        quad_points, quad_weights = self.quadrature(self._quad_degree_reconstruction_source)
        basis, dbasis = self.evaluate_basis(quad_points)
        _, dbasis_rec = self.evaluate_basis(quad_points, self.degree_reconstruction)
            # Note: we're actually evaluating the same basis twice in the above...
        # Drop gradient of constant basis function in reconstruction space
        dbasis_rec = dbasis_rec[:,1:]
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
        faces = np.array([self.x_left, self.x_right])
        faces_normal = np.array([-1,1])
        basis, _ = self.evaluate_basis(faces)
        _, dbasis_rec = self.evaluate_basis(faces, self.degree_reconstruction)
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
        self._reconstruction_source = source

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
        # solve_reconstruction
        rhs = self.reconstruction_source @ dofs
        reconstruction = cho_solve((self.reconstruction_cho, self._reconstruction_cho_lower), rhs, overwrite_b=True)
        # Append the value of the average to complete the DOFs of the reconstruction
        return np.concatenate(([dofs[0]], reconstruction))
