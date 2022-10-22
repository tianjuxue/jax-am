import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d
from jax_am.cfd.json_parser import cfd_parse
from jax_am.phase_field.utils import Field, walltime, read_path
from jax_am.phase_field.yaml_parser import pf_parse
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

jax.config.update("jax_enable_x64", True)


def coupled_integrator():
    """One-way coupling of CFD solver and PF solver.
    Namely, PF solver consumes temperature field produced by CFD solver in each time step.
    """
    def convert_temperature(T_cfd):
        """CFD temperature is (Nx, Ny, Nz), but PF temperature needs (Nz, Ny, Nz)
        """
        T_pf = np.transpose(T_cfd, axes=(2, 1, 0)).reshape(-1, 1)
        return T_pf

    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    pf_args = pf_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir

    generate_neper = False
    if generate_neper:
        pre_processing(pf_args)
 
    polycrystal = Field(pf_args)

    mesh = mesh3d([pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']], 
                  [pf_args['Nx'], pf_args['Ny'],pf_args['Nz']])
    # mesh_local = mesh
    mesh_local = mesh3d([0.75*pf_args['domain_x'], 0.5*pf_args['domain_y'], 0.4*pf_args['domain_z']], 
                        [round(0.75*pf_args['Nx']), round(0.5*pf_args['Ny']), round(0.4*pf_args['Nz'])])

    cfd_args = cfd_parse(os.path.join(crt_file_path, 'cfd_params.json'))
    cfd_args['mesh'] = mesh
    cfd_args['mesh_local'] = mesh_local
    cfd_args['cp'] = lambda T: (0.2441*np.clip(T,300,1563)+338.39) 
    cfd_args['k'] = lambda T: 0.0163105*np.clip(T,300,1563)+4.5847
    cfd_args['data_dir'] = data_dir

    pf_solver = PFSolver(pf_args, polycrystal)
    pf_sol0 = pf_solver.ini_cond()

    cfd_solver = AM_3d(cfd_args)
    
    ts = np.arange(0., cfd_args['laser_path']['time'][-1], cfd_args['dt'])

    pf_solver.clean_sols()

    pf_state = (pf_sol0, ts[0])

    T0 = convert_temperature(cfd_solver.T0)
    pf_params = [T0]
 
    pf_solver.write_sols(pf_sol0, T0, 0)
    for (i, t_crt) in enumerate(ts[1:]):

        pf_state, pf_sol = walltime()(pf_solver.stepper)(pf_state, t_crt, pf_params)
        walltime()(cfd_solver.time_integration)()

        T = convert_temperature(cfd_solver.T0)
        pf_params = [T]

        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T, ts, i + 1)

        if (i + 1) % pf_args['write_sol_interval'] == 0:
            pf_solver.write_sols(pf_sol, T, i + 1)

        if (i + 1) % cfd_args['write_sol_interval'] == 0:
            print(f'T_max:{cfd_solver.T0.max()},vmax:{np.linalg.norm(cfd_solver.vel0,axis=3).max()}')
            # cfd_solver.write_sols(i + 1)
            

if __name__=="__main__":
    coupled_integrator()
