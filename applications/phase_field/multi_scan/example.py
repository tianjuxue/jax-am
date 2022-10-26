import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import sys
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d
from jax_am.phase_field.utils import Field, walltime, process_eta
from jax_am.phase_field.yaml_parser import pf_parse
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)


def set_params():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    pf_args = pf_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir
    return pf_args


def get_T_fn(polycrystal, pf_args):
    def get_T_laser(t, x_laser, y_laser, z_laser, power, unit_mov_dir):
        """Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
        """
        centroids = polycrystal.centroids
        Q = 25
        alpha = 5.2e-6
        kappa = 27
        X = centroids[:, 0] - x_laser
        Y = centroids[:, 1] - y_laser
        Z = centroids[:, 2] - z_laser
        R = np.sqrt(X**2 + Y**2 + Z**2)
        projection = X*unit_mov_dir[0] + Y*unit_mov_dir[1] + Z*unit_mov_dir[2]
        T = pf_args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-pf_args['vel'] / (2*alpha) * (R + projection))
        T = np.where(T > 2000., 2000., T)
        return T[:, None]
    return jax.jit(get_T_laser)


def read_path(pf_args):
    x_corners = pf_args['laser_path']['x_pos']
    y_corners = pf_args['laser_path']['y_pos']
    z_corners = pf_args['laser_path']['z_pos']
    power_control = pf_args['laser_path']['switch'][:-1]

    ts, xs, ys, zs, ps, mov_dir = [], [], [], [], [], []
    t_pre = 0.
    for i in range(len(x_corners) - 1):
        moving_direction = onp.array([x_corners[i + 1] - x_corners[i], 
                                      y_corners[i + 1] - y_corners[i],
                                      z_corners[i + 1] - z_corners[i]])
        traveled_dist = onp.linalg.norm(moving_direction)
        unit_direction = moving_direction/traveled_dist
        traveled_time = traveled_dist/pf_args['vel']
        ts_seg = onp.arange(t_pre, t_pre + traveled_time, pf_args['dt'])
        xs_seg = onp.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = onp.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        zs_seg = onp.linspace(z_corners[i], z_corners[i + 1], len(ts_seg))
        ps_seg = onp.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        zs.append(zs_seg)
        ps.append(ps_seg)
        mov_dir.append(onp.repeat(unit_direction[None, :], len(ts_seg), axis=0))
        t_pre = t_pre + traveled_time

    ts, xs, ys, zs, ps, mov_dir = onp.hstack(ts), onp.hstack(xs), onp.hstack(ys), onp.hstack(zs), onp.hstack(ps), onp.vstack(mov_dir)  
    print(f"Total number of time steps = {len(ts)}")
    combined = onp.hstack((ts[:, None], xs[:, None], ys[:, None], zs[:, None], ps[:, None], mov_dir))
    # print(combined)
    return ts, xs, ys, zs, ps, mov_dir

def generate_neper():
    pf_args = set_params()
    pre_processing(pf_args)

def integrator():
    pf_args = set_params()
    polycrystal = Field(pf_args)
    pf_solver = PFSolver(pf_args, polycrystal)
    pf_sol0 = pf_solver.ini_cond()
    ts, xs, ys, zs, ps, mov_dir = read_path(pf_args)
    pf_solver.clean_sols()
    pf_state = (pf_sol0, ts[0])
    T_laser_fn = get_T_fn(polycrystal, pf_args)
    T0 = T_laser_fn(ts[0], xs[0], ys[0], zs[0], ps[0], mov_dir[0])
    pf_params = [T0]
 
    pf_solver.write_sols(pf_sol0, T0, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        pf_state, pf_sol = pf_solver.stepper(pf_state, t_crt, pf_params)
        T = T_laser_fn(t_crt, xs[i + 1], ys[i + 1], zs[i + 1], ps[i + 1], mov_dir[i + 1])
        pf_params = [T]
        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)
        if (i + 1) % pf_args['write_sol_interval'] == 0:
            pf_solver.write_sols(pf_sol, T, i + 1)


def post_processing():
    pf_args = set_params()
    polycrystal = Field(pf_args)
    process_eta(pf_args)

 
if __name__=="__main__":
    generate_neper()
    integrator()
    post_processing()

