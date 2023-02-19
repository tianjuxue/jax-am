import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d
from jax_am.phase_field.utils import Field, walltime
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing

from jax_am.common import yaml_parse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_T_fn(polycrystal, pf_args):
    def get_T_laser(t):
        """Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
        """
        centroids = polycrystal.centroids
        Q = 25
        alpha = 5.2e-6
        kappa = 27
        x0 = 0.2*pf_args['domain_x']
        vel = 0.6*pf_args['domain_x'] / pf_args['t_OFF']
        X = centroids[:, 0] - x0 - vel * t
        Y = centroids[:, 1] - 0.5*pf_args['domain_y']
        Z = centroids[:, 2] - pf_args['domain_z']
        R = np.sqrt(X**2 + Y**2 + Z**2)
        T = pf_args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-vel / (2*alpha) * (R + X))
        return T[:, None]
    return jax.jit(get_T_laser)


def integrator():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    pf_args = yaml_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir

    pre_processing(pf_args)

    polycrystal = Field(pf_args)
    pf_solver = PFSolver(pf_args, polycrystal)
    pf_sol0 = pf_solver.ini_cond()
    EPS = 1e-10
    ts = np.arange(0., pf_args['t_OFF'] + EPS, pf_args['dt'])
    pf_solver.clean_sols()
    pf_state = (pf_sol0, ts[0])
    T_laser_fn = get_T_fn(polycrystal, pf_args)
    T0 = T_laser_fn(ts[0])
    pf_params = [T0]
 
    pf_solver.write_sols(pf_sol0, T0, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        pf_state, pf_sol = pf_solver.stepper(pf_state, t_crt, pf_params)
        T = T_laser_fn(t_crt)
        pf_params = [T]
        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)
        if (i + 1) % pf_args['write_sol_interval'] == 0:
            pf_solver.write_sols(pf_sol, T, i + 1)


if __name__=="__main__":
    integrator()
