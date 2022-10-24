import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import glob
from functools import partial
from orix.quaternion import Orientation

from jax_am.cfd.cfd_am import mesh3d, AM_3d
from jax_am.phase_field.utils import Field, walltime
from jax_am.phase_field.yaml_parser import pf_parse
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_T_fn(polycrystal, pf_args):
    def get_T_quench(t):
        '''
        Given spatial coordinates and t, we prescribe the value of T.
        '''
        centroids = polycrystal.centroids
        z = centroids[:, 2]
        vel = 0.2
        thermal_grad = 5.e5
        cooling_rate = thermal_grad * vel
        t_total = pf_args['domain_z'] / vel
        T = pf_args['T_liquidus'] + thermal_grad * z - cooling_rate * t
        return T[:, None]
    return jax.jit(get_T_quench)


def get_ori2(pf_args):
    # axes = onp.array([[1., 0., 0.], 
    #                   [1., 1., 0.],
    #                   [1., 1., 1.],
    #                   [1., 1., 0.],
    #                   [1., 0., 0.], 
    #                   [1., -1., 0.]])
    # angles = onp.array([0., 
    #                     onp.pi/8,
    #                     onp.pi/4,
    #                     onp.pi/4, 
    #                     onp.pi/4,
    #                     onp.pi/2 - onp.arccos(onp.sqrt(2)/onp.sqrt(3))])

    axes = onp.array([[1., 0., 0.], 
                      [1., 0., 0.], 
                      [1., -1., 0.]])
    angles = onp.array([0., 
                        onp.pi/4,
                        onp.pi/2 - onp.arccos(onp.sqrt(2)/onp.sqrt(3))])

    assert pf_args['num_oris'] == len(axes), f"Set num_oris = {pf_args['num_oris']}, but actually {len(axes)}"
    ori2 = Orientation.from_axes_angles(axes, angles)
    return ori2


def integrator():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    pf_args = pf_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir
    generate_neper = False
    if generate_neper:
        pre_processing(pf_args)

    ori2 = get_ori2(pf_args)
    polycrystal = Field(pf_args, ori2)
    pf_solver = PFSolver(pf_args, polycrystal)
    pf_sol0 = pf_solver.ini_cond()
    ts = np.arange(0., pf_args['t_OFF'] + 1e-10, pf_args['dt'])
    pf_solver.clean_sols()
    pf_state = (pf_sol0, ts[0])
    T_quench_fn = get_T_fn(polycrystal, pf_args)
    T0 = T_quench_fn(ts[0])
    pf_params = [T0]
 
    pf_solver.write_sols(pf_sol0, T0, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        pf_state, pf_sol = pf_solver.stepper(pf_state, t_crt, pf_params)
        T = T_quench_fn(t_crt)
        pf_params = [T]
        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)
        if (i + 1) % pf_args['write_sol_interval'] == 0:
            pf_solver.write_sols(pf_sol, T, i + 1)


if __name__=="__main__":
    integrator()
