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
                    raise ValueError("Neumann system requires specification of the average to be solvable.")
            case _:
                if not average is None:
                     raise ValueError(f"Average of the solution cannot be imposed with boundary conditions [{self.boundary_conditions}]")
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

        Warns
        -----
            RuntimeWarning
                If the boundary conditions described by bc_left, bc_right and average do not correspond to self.boundary_conditions settings.
        """
        self.solve_transmission(f, bc_left, bc_right, average)
        self.solve_cell_problems()


    def plot(self, ax):
        """
        Plot solution at the faces and on the cells on the provided Matplotlib axis.

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
            plt_i, = ax.plot(cell.points_plot, 
                    cell.solution_plot, 
                    color='#1f77b4', 
                    linewidth=2)
            if i == 0:
                plt_i.set_label("HHO \u2014 cells")
        ax.set_title(f"Approximation by HHO method (degree {self.cell_degree})")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.legend()


class HHO_cell:    
    """
    Handles all information regarding local HHO problems on a single cell.

    Attributes
    ----------
        x_left : double
            Left vertex of the cell.
        x_right : double
            Right vertex of the cell.
        h : double
            length of the cell.
        degree : int
            Polynomial degree of the cell unknowns.
        solution : ndarray
            Solution to the local problem on the cell in the polynomial basis, computed by self.solve()
        plot_margin : ndarray
            Margin around the faces that is used for visualization of the discontinuous solution.
        points_plot : ndarray
            Points in the cell used to plot the cell solution.
        solution_plot : ndarray
            Representation of the solution in the cell on points_plot.

    Methods
    -------
        solve(source, sol_global_left, sol_global_right))
            Solve local problem based on the source term and the solution to the global solution on both faces of the cell.
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
        self.h = x_right - self.x_left
        self.degree = degree

        self._solution = None

        self.plot_margin = self.h * 1E-5
        self._points_plot = None
        self._solution_plot = None

    @property 
    def solution(self):
        """
        Solution to the local problem on the cell in the polynomial basis.

        Can only be set through the solve method.
        """
        assert self._solution is not None, "Solution to local problem has not yet been computed."
        return self._solution

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
        if self.degree == 0:
            source_average = self.h * (source(self.x_left) + source(self.x_right))/2.0
            solution = 0.5 * self.h**2 * source_average + 0.5*(sol_global_left + sol_global_right)
        else:
            raise ValueError("Local problems have only been implemented for cell degree 0.")
        self._solution = solution
        self._solution_plot = None


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
    
