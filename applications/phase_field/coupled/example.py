import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d

from jax_am.phase_field.utils import Field
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing

from jax_am.common import box_mesh, json_parse, yaml_parse, walltime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

jax.config.update("jax_enable_x64", True)


def coupled_integrator():
    """One-way coupling of CFD solver and PF solver.
    Namely, PF solver consumes temperature field produced by CFD solver in each time step.

    TODO: 
    (1) Multi-scan tool path
    (2) Spatial interpolation
    """
    @jax.jit
    def convert_temperature(T_cfd):
        """CFD temperature is (Nx, Ny, Nz), but PF temperature needs (Nz, Ny, Nx)
        """
        T_pf = np.transpose(T_cfd, axes=(2, 1, 0)).reshape(-1, 1)
        return T_pf

    @jax.jit
    def interpolate_T(T_past, T_future, t_pf, t_cfd):
        ratio = (t_cfd - t_pf)/cfd_args['dt']
        T = ratio*T_past + (1 - ratio)*T_future
        return T

    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    pf_args = yaml_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir

    pre_processing(pf_args)
 
    polycrystal = Field(pf_args)

    mesh = mesh3d([pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']], 
                  [pf_args['Nx'], pf_args['Ny'], pf_args['Nz']])
    
    Nx_local = round(0.75*pf_args['Nx'])
    Ny_local = round(0.5*pf_args['Ny'])
    Nz_local = round(0.5*pf_args['Nz'])
    mesh_local = mesh3d([Nx_local/pf_args['Nx']*pf_args['domain_x'],
                         Ny_local/pf_args['Ny']*pf_args['domain_y'],
                         Nz_local/pf_args['Nz']*pf_args['domain_z']], 
                    [Nx_local,Ny_local,Nz_local])
    
    meshio_mesh = box_mesh(pf_args['Nx'], pf_args['Ny'], pf_args['Nz'], pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z'])

    cfd_args = json_parse(os.path.join(crt_file_path, 'cfd_params.json'))
    cfd_args['mesh'] = mesh
    cfd_args['mesh_local'] = mesh_local
    cfd_args['cp'] = lambda T: (0.2441*np.clip(T,300,1563)+338.39) 
    cfd_args['k'] = lambda T: 0.0163105*np.clip(T,300,1563)+4.5847
    cfd_args['data_dir'] = data_dir
    cfd_args['meshio_mesh'] = meshio_mesh
    assert cfd_args['dt'] >= pf_args['dt'], f"CFD time step must be greater than PF for intepolation"

    pf_solver = PFSolver(pf_args, polycrystal)
    pf_sol0 = pf_solver.ini_cond()
    pf_ts = np.arange(0., pf_args['t_OFF'] + 1e-10, pf_args['dt'])
    t_pf = pf_ts[0]
    pf_state = (pf_sol0, t_pf)

    cfd_solver = AM_3d(cfd_args)
    cfd_ts = np.arange(0., cfd_args['t_OFF'] + 1e-10, cfd_args['dt'])
    cfd_solver.clean_sols()
    cfd_step = 0
    cfd_solver.write_sols(cfd_step)    
    T_past = cfd_solver.T[:,:,:,0]
    walltime()(cfd_solver.time_integration)()

    T_future = cfd_solver.T[:,:,:,0]
    t_cfd = cfd_args['dt']
    cfd_step += 1

    T_pf = convert_temperature(interpolate_T(T_past, T_future, t_pf, t_cfd))
    pf_solver.write_sols(pf_sol0, T_pf, 0)

    for (i, t_pf) in enumerate(pf_ts[1:]):
        # Assume that t_cfd < t_pf <= t_cfd + cfd_args['dt']
        if t_pf > t_cfd + cfd_args['dt']:
            walltime()(cfd_solver.time_integration)()
            T_past = T_future
            T_future = cfd_solver.T[:,:,:,0]
            t_cfd += cfd_args['dt']
            cfd_step += 1
            if cfd_step % cfd_args['check_sol_interval'] == 0:
                cfd_solver.inspect_sol(cfd_step, len(cfd_ts[1:]))

            if cfd_step % cfd_args['write_sol_interval'] == 0:
                cfd_solver.write_sols(cfd_step)

        T_pf = convert_temperature(interpolate_T(T_past, T_future, t_pf, t_cfd))
        pf_state, pf_sol = walltime()(pf_solver.stepper)(pf_state, t_pf, [T_pf])
        
        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T_pf, pf_ts, i + 1)

        if (i + 1) % pf_args['write_sol_interval'] == 0:
            pf_solver.write_sols(pf_sol, convert_temperature(T_future), i + 1)


if __name__=="__main__":
    coupled_integrator()
