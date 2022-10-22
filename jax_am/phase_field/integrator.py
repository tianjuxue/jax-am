# import jax
# import jax.numpy as np
# import numpy as onp
# from functools import partial

# # from modules.phase_field.yaml_parse import args
# from modules.phase_field.utils import read_path, walltime, Field
# from modules.phase_field.allen_cahn import PFSolver
# from modules.phase_field.cfd import CFDSolver

# from modules.cfd.cfd_am import AM_3d
# from modules.cfd.parser import cfd_parse


# class MultiVarSolver:
#     '''
#     One-way coupling of CFD solver and PF solver.
#     Namely, PF solver consumes temperature field produced by CFD solver in each time step.
#     '''
#     def __init__(self, T_fn):
#         self.polycrystal = Field()
#         self.T_fn = T_fn

#     @walltime
#     def solve(self):

#         pf_solver = PFSolver(self.polycrystal)
#         pf_sol0 = pf_solver.ini_cond()

#         cfd_solver = CFDSolver(self.polycrystal, self.T_fn)
#         cfd_sol0 = cfd_solver.ini_cond()

#         # TODO: We only need ts here, perhaps even don't need ts.
#         ts, xs, ys, ps = read_path()

#         pf_solver.clean_sols()

#         pf_state = (pf_sol0, ts[0])
#         cfd_state = (cfd_sol0, ts[0])

#         T0 = cfd_sol0[:, 0:1]
#         pf_params = [T0]
#         cfd_params = []

#         pf_solver.write_sols(pf_sol0, T0, 0)
#         for (i, t_crt) in enumerate(ts[1:]):
#             pf_state, pf_sol = pf_solver.stepper(pf_state, t_crt, pf_params)
#             cfd_state, cfd_sol = cfd_solver.stepper(cfd_state, t_crt, cfd_params)

#             T = cfd_sol[:, 0:1]
#             pf_params = [T]

#             if (i + 1) % args['check_sol_interval'] == 0:
#                 pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)

#             if (i + 1) % args['write_sol_interval'] == 0:
#                 pf_solver.write_sols(pf_sol, T, i + 1)

#         pf_solver.polycrystal.write_info()



# def pf_integrator(data_dir, T_fn):
#     # TODO: We only need ts here, perhaps even don't need ts.
#     ts, xs, ys, ps = read_path()



