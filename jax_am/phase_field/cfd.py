# import jax
# import jax.numpy as np
# from functools import partial
# from modules.phase_field.yaml_parse import args
# from modules.phase_field.abstract_solver import ODESolver
# from modules.phase_field.utils import read_path


# class CFDSolver(ODESolver):
#     def __init__(self, polycrystal, T_fn):
#         super().__init__(polycrystal)
#         self.T_fn = T_fn


#     def stepper(self, state_pre, t_crt, ode_params):
#         T = self.T_fn(t_crt, self.polycrystal)
#         return (T, t_crt), T
        

#     def ini_cond(self):
#         T0 = self.T_fn(0., self.polycrystal)
#         return T0


# ########################################################################################################################
# # Remark(Tianju): Shuheng, you can start from here.

# class CFDSolverToBeImplemented(ODESolver):
#     def __init__(self, polycrystal):
#         super().__init__(polycrystal)
#         # For spatial discretization, you will find self.polycrystal.mesh useful
#         # self.polycrystal.mesh is a meshio object. 
#         # You may print mesh.points (all vertices) and mesh.cells_dict['hexahedron'] (all cells) for more information.

#         # For temporal discretization and laser path, we can try to find a suitable time step for both CFD and PF.
#         # Currently, you can refer to the function ts, xs, ys, ps = read_path() in utils.py

#     def stepper(self, state_pre, t_crt, ode_params):
#         # You may refer to the class PFSolver in allen_cahn.py to see how this function should be implemented.
#         pass
        

#     def ini_cond(self):
#         # You may refer to the class PFSolver in allen_cahn.py to see how this function should be implemented.
#         pass
