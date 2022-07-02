import jax
import jax.numpy as np
from functools import partial
from src.yaml_parse import args
from src.abstrac_solver import ODESolver
from src.utils import read_path


@partial(jax.jit, static_argnums=(1,))
def get_T(t, polycrystal):
    '''
    Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
    '''
    centroids = polycrystal.centroids

    Q = 25
    alpha = 5.2
    kappa = 2.7*1e-2
    x0 = 0.2*args['domain_x']
    vel = 0.6*args['domain_x'] / args['laser_time']['time'][-1]

    X = centroids[:, 0] - x0 - vel * t
    Y = centroids[:, 1] - 0.5*args['domain_y']
    Z = centroids[:, 2] - args['domain_z']
    R = np.sqrt(X**2 + Y**2 + Z**2)
    T = args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-vel / (2*alpha) * (R + X))

    # TODO: Not quite elegant here
    T = np.where(T > 2000., 2000., T)

    return T[:, None]


class CFDSolver(ODESolver):
    def __init__(self, polycrystal):
        super().__init__(polycrystal)


    def stepper(self, state_pre, t_crt, ode_params):
        T = get_T(t_crt, self.polycrystal)
        return (T, t_crt), T
        

    def ini_cond(self):
        T0 = get_T(0., self.polycrystal)
        return T0


########################################################################################################################
# Remark(Tianju): Shuheng, you can start from here.

class CFDSolverToBeImplemented(ODESolver):
    def __init__(self, polycrystal):
        super().__init__(polycrystal)
        # For spatial discretization, you will find self.polycrystal.mesh useful
        # self.polycrystal.mesh is a meshio object. 
        # You may print mesh.points (all vertices) and mesh.cells_dict['hexahedron'] (all cells) for more information.

        # For temporal discretization and laser path, we can try to find a suitable time step for both CFD and PF.
        # Currently, you can refer to the function ts, xs, ys, ps = read_path() in utils.py

    def stepper(self, state_pre, t_crt, ode_params):
        # You may refer to the class PFSolver in allen_cahn.py to see how this function should be implemented.
        T = get_T(t_crt, self.polycrystal)
        return (T, t_crt), T
        

    def ini_cond(self):
        # You may refer to the class PFSolver in allen_cahn.py to see how this function should be implemented.
        T0 = get_T(0., self.polycrystal)
        return T0
