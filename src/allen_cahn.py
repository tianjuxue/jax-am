import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import time
from functools import partial
from src.yaml_parse import args
from src.abstract_solver import ODESolver

# gpus = jax.devices('gpu')
# @partial(jax.jit, static_argnums=(2,), device=gpus[-1])


def phase_field(polycrystal):

    # TODO: make this simpler
    mesh_h = polycrystal.ch_len[0]

    # TODO: consider anisotropic growth
    def update_anisotropy():
        centroids = polycrystal.centroids

        sender_centroids = np.take(centroids, graph.senders, axis=0)
        receiver_centroids = np.take(centroids, graph.receivers, axis=0)
        edge_directions = sender_centroids - receiver_centroids
        edge_directions = np.repeat(edge_directions[:, None, :], args['num_oris'], axis=1) # (num_edges, num_oris, dim)
 
        unique_grain_directions = polycrystal.unique_grain_directions # (num_directions_per_cube, num_oris, dim)

        assert edge_directions.shape == (len(graph.senders), args['num_oris'], args['dim'])
        cosines = np.sum(unique_grain_directions[None, :, :, :] * edge_directions[:, None, :, :], axis=-1) \
                  / (np.linalg.norm(edge_directions, axis=-1)[:, None, :])
        anlges = np.arccos(cosines) 
        anlges = np.where(np.isfinite(anlges), anlges, 0.)
        anlges = np.where(anlges < np.pi/2., anlges, np.pi - anlges)
        anlges = np.min(anlges, axis=1)

        anisotropy_term = 1. + args['anisotropy'] * (np.cos(anlges)**4 + np.sin(anlges)**4) # (num_edges, num_oris)

        assert anisotropy_term.shape == (len(graph.senders), args['num_oris'])
        graph.edges['anisotropy'] = anisotropy_term
        print("End of compute_anisotropy...")

    def local_energy_fn(eta, zeta):
        gamma = 1
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.))
        graph_energy_2 = gamma * (np.sum(vmap_outer(eta, eta)**2) - np.sum(eta**4))  
        graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1).reshape(-1))
        grain_energy = args['m_g'] * (grain_energy_1 +  graph_energy_2 + graph_energy_3)
        return grain_energy

    local_energy_grad_fn = jax.grad(local_energy_fn, argnums=0) 

    def state_rhs(state, t, ode_params):
        eta = state

        T, = ode_params

        zeta = 0.5 * (1 - np.tanh(1e1*(T/args['T_melt'] - 1)))
        local_energy_grad = local_energy_grad_fn(eta, zeta) / args['ad_hoc']
        # Make the code concise
        # https://github.com/google/jax-cfd/blob/8eff9c47bdc7fb19b6453db94ca65f6be64d91f6/jax_cfd/base/finite_differences.py#L74
        eta_xyz = np.reshape(eta, (args['Nz'], args['Ny'], args['Nx'], args['num_oris']))
        eta_neg_x = np.concatenate((eta_xyz[:, :, :1, :], eta_xyz[:, :, :-1, :]), axis=2)
        eta_pos_x = np.concatenate((eta_xyz[:, :, 1:, :], eta_xyz[:, :, -1:, :]), axis=2)
        eta_neg_y = np.concatenate((eta_xyz[:, :1, :, :], eta_xyz[:, :-1, :, :]), axis=1)
        eta_pos_y = np.concatenate((eta_xyz[:, 1:, :, :], eta_xyz[:, -1:, :, :]), axis=1)
        eta_neg_z = np.concatenate((eta_xyz[:1, :, :, :], eta_xyz[:-1, :, :, :]), axis=0)
        eta_pos_z = np.concatenate((eta_xyz[1:, :, :, :], eta_xyz[-1:, :, :, :]), axis=0)
        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -np.stack((eta_pos_x - 2*eta_xyz + eta_neg_x, 
                                 eta_pos_y - 2*eta_xyz + eta_neg_y, 
                                 eta_pos_z - 2*eta_xyz + eta_neg_z), axis=-1) / mesh_h**2 * args['kappa_g'] * args['ad_hoc']
        assert laplace_xyz.shape == (args['Nz'], args['Ny'], args['Nx'], args['num_oris'], args['dim'])
        laplace = np.sum(laplace_xyz.reshape(-1, args['num_oris'], args['dim']), axis=-1)
        assert local_energy_grad.shape == laplace.shape
        Lg = args['L0'] * np.exp(-args['Qg'] / (T*args['gas_const']))
        rhs = -Lg * (local_energy_grad + laplace)
        return rhs

    return state_rhs


@partial(jax.jit, static_argnums=(2,))
def rk4(state_pre, t_crt, f, ode_params):
    '''
    Fourth order Runge-Kutta method
    We probably don't need this one.
    '''
    y_prev, t_prev = state_pre
    h = t_crt - t_prev
    k1 = h * f(y_prev, t_prev, ode_params)
    k2 = h * f(y_prev + k1/2., t_prev + h/2., ode_params)
    k3 = h * f(y_prev + k2/2., t_prev + h/2., ode_params)
    k4 = h * f(y_prev + k3, t_prev + h, ode_params)
    y_crt = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def explicit_euler(state_pre, t_crt, f, ode_params):
    '''
    Explict Euler method
    '''
    y_prev, t_prev = state_pre
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, ode_params)
    return (y_crt, t_crt), y_crt


class PFSolver(ODESolver):
    def __init__(self, polycrystal):
        super().__init__(polycrystal)
        self.state_rhs = phase_field(self.polycrystal)


    def stepper(self, state_pre, t_crt, ode_params):
        return explicit_euler(state_pre, t_crt, self.state_rhs, ode_params)
        

    def ini_cond(self):
        '''
        Prescribe the initial conditions for eta.
        '''
        num_nodes = len(self.polycrystal.centroids)
        eta = np.zeros((num_nodes, args['num_oris']))
        eta = eta.at[np.arange(num_nodes), self.polycrystal.cell_ori_inds].set(1)
        y0 = eta
        return y0


    def clean_sols(self):
        '''
        Clean the data folder.
        '''

        # TODO: create the folder if not exist! Important!

        vtk_folder = f"post-processing/vtk/{args['case']}/pf/sols"
        numpy_folder = f"post-processing/numpy/{args['case']}/pf/sols"
        files_vtk = glob.glob(vtk_folder + f"/*")
        files_numpy = glob.glob(numpy_folder + f"/*")
        files = files_vtk + files_numpy
        for f in files:
            os.remove(f)


    def inspect_sol(self, pf_sol, pf_sol0, T, ts, step):
        '''
        While running simulations, print out some useful information.
        '''
        # print(pf_sol[:10, :5])
        print(f"step {step} of {len(ts[1:])}, unix timestamp = {time.time()}")
        eta0 = np.argmax(pf_sol0, axis=1)
        eta = np.argmax(pf_sol, axis=1)
        change_eta = np.where(eta0 == eta, 0, 1)
        change_T = np.where(T >= args['T_melt'], 1, 0)
        print(f"percent of change of orientations = {np.sum(change_eta)/len(change_eta)*100}%")
        print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
        print(f"max T = {np.max(T)}")
 
        if not np.all(np.isfinite(pf_sol)):          
            raise ValueError(f"Found np.inf or np.nan in pf_sol - stop the program")


    def write_sols(self, pf_sol, T, step):
        '''
        Output solution .vtu files that can be visualized by Paraview: 
        1. Temeperature field (:math:`T`)
        2. Liquid/Solid phase (:math:`\zeta`)
        3. Grain orientations (:math:`\eta`)
        '''
        print(f"Write sols to file...")

        step = (step + 1) // args['write_sol_interval']
 
        zeta = T.reshape(-1) < args['T_melt']
        eta = pf_sol
        eta_max = onp.max(eta, axis=1)
        cell_ori_inds = onp.argmax(eta, axis=1)
        ipf_x = onp.take(self.polycrystal.unique_oris_rgb[0], cell_ori_inds, axis=0)
        ipf_y = onp.take(self.polycrystal.unique_oris_rgb[1], cell_ori_inds, axis=0)
        ipf_z = onp.take(self.polycrystal.unique_oris_rgb[2], cell_ori_inds, axis=0)

        # TODO: Is this better?
        ipf_x[zeta < 0.1] = 0.
        ipf_y[zeta < 0.1] = 0.
        ipf_z[zeta < 0.1] = 0.

        self.polycrystal.mesh.cell_data['T'] = [onp.array(T, dtype=onp.float32)]
        self.polycrystal.mesh.cell_data['ipf_x'] = [ipf_x]
        self.polycrystal.mesh.cell_data['ipf_y'] = [ipf_y]
        self.polycrystal.mesh.cell_data['ipf_z'] = [ipf_z]
        cell_ori_inds = onp.array(cell_ori_inds, dtype=onp.int32)
        self.polycrystal.mesh.cell_data['ori_inds'] = [cell_ori_inds]

        onp.save(f"post-processing/numpy/{args['case']}/pf/sols/T_{step:03d}.npy", T)
        onp.save(f"post-processing/numpy/{args['case']}/pf/sols/cell_ori_inds_{step:03d}.npy", cell_ori_inds)
        self.polycrystal.mesh.write(f"post-processing/vtk/{args['case']}/pf/sols/u{step:03d}.vtu")
