import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from src.yaml_parse import args
from src.utils import read_path, walltime, Field
from src.allen_cahn import PFSolver
from src.cfd import CFDSolver


class MultiVarSolver:
    '''
    One-way coupling of CFD solver and PF solver.
    Namely, PF solver consumes temperature field produced by CFD solver in each time step.
    '''
    def __init__(self, toolpath_file):
        self.toolpath_file = toolpath_file
        self.polycrystal = Field()

    @walltime
    def solve(self):

        pf_solver = PFSolver(self.polycrystal)
        pf_sol0 = pf_solver.ini_cond()

        cfd_solver = CFDSolver(self.polycrystal)
        cfd_sol0 = cfd_solver.ini_cond()

        # TODO: We only need ts here, perhaps even don't need ts.
        ts, xs, ys, ps = read_path()
  

        pf_solver.clean_sols()

        pf_state = (pf_sol0, ts[0])
        cfd_state = (cfd_sol0, ts[0])

        T0 = cfd_sol0[:, 0:1]
        pf_params = [T0]
        cfd_params = []

        pf_solver.write_sols(pf_sol0, T0, 0)
        for (i, t_crt) in enumerate(ts[1:]):
            pf_state, pf_sol = pf_solver.stepper(pf_state, t_crt, pf_params)
            cfd_state, cfd_sol = cfd_solver.stepper(cfd_state, t_crt, cfd_params)

            T = cfd_sol[:, 0:1]
            pf_params = [T]

            if (i + 1) % args['check_sol_interval'] == 0:
                pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)

            if (i + 1) % args['write_sol_interval'] == 0:
                pf_solver.write_sols(pf_sol, T, i + 1)

        pf_solver.polycrystal.write_info()
