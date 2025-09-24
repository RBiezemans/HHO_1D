import matplotlib.pyplot as plt
import numpy as np 

from HHO_kernel import HHO_kernel, HHO_cell

def plot_basis(pcell: HHO_cell, plot_xx: np.ndarray):
    """
    Plot basis if pcell on the grid plot_xx.

    Parameters
    ----------
        pcell: HHO_cell
            Cell to perform the selected computation on.
        plot_xx: ndarray
            Grid used for plotting the results on the cell.
    """
    # Evaluate basis on the prototypical cell
    basis, gradient = pcell.evaluate_basis(plot_xx)

    # Plot the results
    fig, axs = plt.subplots(1, 2,
                            figsize=(10,5))
    for kk in range(pcell.degree+1):
        label = rf"$\phi_{kk} \sim x^{kk}$"
        axs[0].plot(plot_xx, basis[:,kk], label=label)
        axs[1].plot(plot_xx, gradient[:,kk])
    for ax in axs:
        ax.tick_params(direction='in')
        ax.set_xlabel('x')
    axs[0].set_title("Basis functions")
    axs[1].set_title("Derivative")
    axs[0].set_ylabel(r"$\phi(x)$")
    axs[1].set_ylabel(r"$\phi'(x)$")
    axs[0].legend()
    fig.suptitle(f"{pcell.basis_type.capitalize()} basis")

def test_HHO_computation(computation: str, pcell: HHO_cell, plot_xx: np.ndarray):
    """
    Illustrate the result of several stages of computations in the local problem on pcell.
    
    Parameters
    ----------
        computation: str
            "L2 projection" / "Reconstruction" / "Cell problem"
        pcell: HHO_cell
            Cell to perform the selected computation on.
        plot_xx: ndarray
            Grid used for plotting the results on the cell.
    """
    xL = pcell.x_left
    xR = pcell.x_right
    # Define test functions for illustration
    f = [lambda x: (x-xL)**5 - (x-xL)*(x-xR),
         lambda x: x*(x-xL)**5 - (x-xL)*(x-xR),
         lambda x: np.cos(1.8*np.pi/(xR-xL)*(x-xL))]
    neg_ddf = [lambda x: -20*(x-xL)**3 + 2,
               lambda x: -20*x*(x-xL)**3 - 10*(x-xL)**4 + 2,
               lambda x: (1.8*np.pi/(xR-xL))**2 * np.cos(1.8*np.pi/(xR-xL)*(x-xL))]
    f_titles = ["5th order polynomial",
                "6th order polynomial",
                "Cosine"]
    
    # Function to compute an HHO result of a function f and plot it on the axis ax along with the exact function f
    def plot_HHO_computation(f, neg_ddf, ax, computation):
        # Compute HHO result according to required input
        match computation:
            case "L2 projection":
                f_HHO, _ = pcell.evaluate_fun(plot_xx, pcell.compute_L2_projection(f), "basis")
                label = "f"
            case "Reconstruction":
                dofs = np.concatenate((pcell.compute_L2_projection(f), [f(xL), f(xR)]))
                f_HHO, _ = pcell.cell_reconstruction.evaluate_fun(plot_xx, pcell.compute_reconstruction(dofs), "basis")
                label = "u"
            case "Cell problem":
                pcell.solve(neg_ddf, f(xL), f(xR))
                dofs = np.concatenate((pcell.solution, pcell.solution_faces))
                f_HHO, _ = pcell.cell_reconstruction.evaluate_fun(plot_xx, pcell.compute_reconstruction(dofs), "basis")
                label = "u"
        # Plot f
        ax.plot(plot_xx,f(plot_xx),'r.',label=label)
        # Compute error in the max norm
        err_max = np.max(np.abs( f(plot_xx) - f_HHO ))
        # Plot the projection on the cell
        label = f"{computation}\nMax error: {err_max:.1e}"
        ax.plot(plot_xx,f_HHO,'b-',label=label)
        # Compute max error at the faces
        err_max_faces = np.max(np.abs(([f(plot_xx[0]) - f_HHO[0], f(plot_xx[-1]) - f_HHO[-1]])))
        if computation in ["Reconstruction", "Cell problem"]:
            print(f"Max error at the faces: {err_max_faces:.1e} ({f_titles[i]})")

    fig, axs = plt.subplots(len(f), 1 ,
                            figsize=(5,3.5*len(f)),
                            gridspec_kw={'hspace': 0.4})
    for i,ax in enumerate(axs):    
        plot_HHO_computation(f[i], neg_ddf[i], ax, computation)
        ax.legend(loc='center left',
                bbox_to_anchor=(1, 0.5))
        ax.tick_params(direction='in')
        ax.grid(True, 
                which='major', 
                linestyle='--', 
                linewidth=0.5, 
                color='lightgrey')
        axs[i].set_title(f_titles[i])
    
    fig.suptitle(f"Example of {computation}s")

def test_HHO_convergence(computation, test_degrees=[0], basis="monomial"):
    """
    Make convergence curves for different computations on the cell.
    
    Parameters
    ----------
        computation: str
            "L2 projection" / "Reconstruction" / "Poisson solve"
        test_degree: list[int]
            Cell degrees of the HHO method to compute convergence curves for.
        basis: str
            Description of the basis used on the cell.
    """
    x_left = 0
    x_right = 1
    N_cells_min = 2
    N_cells_per_degree = [np.array([2**(max((7-test_degree),N_cells_min)+k) for k in range(6)]) for test_degree in test_degrees]
    f = lambda x : np.cos(4*np.pi*x)
    df = lambda x : -4*np.pi*np.sin(4*np.pi*x)
    neg_ddf = lambda x : (4*np.pi)**2 * np.cos(4*np.pi*x)

    if len(test_degrees) == 1:
        fig_rows = 1
        fig_cols = 2
    else:
        fig_rows = 2
        fig_cols = len(test_degrees) 
    fig, axs = plt.subplots(fig_rows, fig_cols, 
                            figsize=(min(12,3*fig_cols),4*fig_rows),
                            gridspec_kw={'hspace': 0.3})
    # Compute L2 and H1 errors for all test degrees
    for (ik, k_test) in enumerate(test_degrees):
        N_cells = N_cells_per_degree[ik]
        errors_L2 = np.zeros(len(N_cells))
        errors_H1 = np.zeros(len(N_cells))
        for (i,N) in enumerate(N_cells):
            grid_N = np.linspace(x_left, x_right, N+1)
            solver = HHO_kernel(grid_N, k_test, basis)
            error_L2_norm = 0
            error_H1_norm = 0
            if computation == "Poisson solve":
                solver.boundary_conditions = 'DD'
                solver.solve(neg_ddf, f(x_left), f(x_right)) 
            for c in solver.cells:
                # Define quadrature to compute the error
                quad_error_x, quad_error_w = c.quadrature((c.degree+2))
                # Compute requested HHO operation
                match computation:
                    case "L2 projection":
                        f_basis, f_gradient = c.evaluate_fun(quad_error_x, c.compute_L2_projection(f))
                    case "Reconstruction" | "Poisson solve":
                        match computation:
                            case "Reconstruction":
                                # Compute DOFs of f in the discrete HHO space on the cell
                                dofs = np.concatenate((c.compute_L2_projection(f), [f(c.x_left), f(c.x_right)]))
                            case "Poisson solve":
                                dofs = np.concatenate((c.solution, c.solution_faces))
                        f_basis, f_gradient = c.cell_reconstruction.evaluate_fun(quad_error_x, c.compute_reconstruction(dofs))
                # Compute norm of error
                error_L2_x = f_basis - f(quad_error_x)
                error_H1_x = f_gradient - df(quad_error_x)
                error_L2_norm += np.dot(quad_error_w, error_L2_x**2)
                error_H1_norm += np.dot(quad_error_w, error_H1_x**2) + error_L2_norm
            errors_L2[i] = np.sqrt(error_L2_norm)
            errors_H1[i] = np.sqrt(error_H1_norm)
        # Plot L2 and H1 errors for given test degree
        for inorm, norm in enumerate(["L2", "H1"]):
            if len(test_degrees)==1:
                ax = axs[inorm]
            else:
                ax = axs[inorm][ik]
            match norm:
                case "L2":
                    errors = errors_L2
                case "H1":
                    errors = errors_H1
            match computation:
                case "L2 projection":
                    match norm:
                        case "L2":
                            convergence_order_theory = k_test+1
                            label = r"$\|| f - \Pi^k_H(f) ||_{L^2(\Omega)}$"
                        case "H1":
                            convergence_order_theory = k_test
                            label = r"$\|| f - \Pi^k_H(f) ||_{H^1(\Omega)}$"
                case "Reconstruction":
                    match norm:
                        case "L2":
                            convergence_order_theory = k_test+2
                            label = r"$\|| u - R_H(\hat{\Pi}^k_H u) ||_{L^2(\Omega)}$"
                        case "H1":
                            convergence_order_theory = k_test+1
                            label = r"$\|| u - R_H(\hat{\Pi}^k_H u) ||_{H^1(\Omega)}$"
                case "Poisson solve":
                    match norm:
                        case "L2":
                            convergence_order_theory = k_test+2
                            label = r"$\|| u - u_H ||_{L^2(\Omega)}$"
                        case "H1":
                            convergence_order_theory = k_test+1
                            label = r"$\|| u - u_H ||_{H^1(\Omega)}$"
            # Plot errors
            ax.plot(1.0/N_cells, errors, '*-', linewidth=0.8, label=label)
            # Compute straight line that indicates the theoretical convergence order
            convergence_line = (1.0/N_cells)**(convergence_order_theory)
            # Rescale the straight line to ligh slightly above the first point of the errors
            convergence_line = convergence_line * 6 / convergence_line[0] * errors[0]
            # Plot layout
            ax.plot(1.0/N_cells, 
                    convergence_line, 
                    label=fr"$Ch^{convergence_order_theory}$")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('$h$')
            ax.grid(True, 
                        which='major', 
                        linestyle='--', 
                        linewidth=0.5, 
                        color='lightgrey')
            ax.set_title(f"k={k_test}")
            ax.legend(loc='upper left')
    fig.suptitle(f"Convergence rate of {computation}")