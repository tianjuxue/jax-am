import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import time
from functools import partial


def phase_field(polycrystal, pf_args):
    # TODO: make this simpler
    h_x, h_y, h_z = polycrystal.mesh_h_xyz

    def update_anisotropy_helper(edges):
        edge_directions = edges.reshape(-1, pf_args['dim'])
        edge_directions = np.repeat(edge_directions[:, None, :], pf_args['num_oris'], axis=1) # (num_edges, num_oris, dim)
        unique_grain_directions = polycrystal.unique_grain_directions # (num_directions_per_cube, num_oris, dim)
        cosines = np.sum(unique_grain_directions[None, :, :, :] * edge_directions[:, None, :, :], axis=-1) \
                  / (np.linalg.norm(edge_directions, axis=-1)[:, None, :])
        angles = np.arccos(cosines) 
        angles = np.where(np.isfinite(angles), angles, 0.)
        angles = np.where(angles < np.pi/2., angles, np.pi - angles)
        angles = np.min(angles, axis=1)
        anisotropy_term = 1. + pf_args['anisotropy'] * (np.cos(angles)**4 + np.sin(angles)**4) # (num_edges, num_oris)
        anisotropy_term = anisotropy_term.reshape((edges.shape[0], edges.shape[1], edges.shape[2], pf_args['num_oris']))
        return anisotropy_term

    def local_energy_fn(eta, zeta):
        gamma = 1
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.))
        graph_energy_2 = gamma * (np.sum(vmap_outer(eta, eta)**2) - np.sum(eta**4))  
        graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1).reshape(-1))
        grain_energy = pf_args['m_g'] * (grain_energy_1 +  graph_energy_2 + graph_energy_3)
        return grain_energy

    local_energy_grad_fn = jax.grad(local_energy_fn, argnums=0) 

    def state_rhs(state, t, ode_params):
        eta = state
        T, = ode_params
        T = np.where(T > 2000., 2000., T)
        zeta = 0.5 * (1 - np.tanh(1e10*(T/pf_args['T_melt'] - 1)))
        local_energy_grad = local_energy_grad_fn(eta, zeta) / pf_args['ad_hoc']

        eta_xyz = np.reshape(eta, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx'], pf_args['num_oris']))
        eta_neg_x = np.concatenate((eta_xyz[:, :, :1, :], eta_xyz[:, :, :-1, :]), axis=2)
        eta_pos_x = np.concatenate((eta_xyz[:, :, 1:, :], eta_xyz[:, :, -1:, :]), axis=2)
        eta_neg_y = np.concatenate((eta_xyz[:, :1, :, :], eta_xyz[:, :-1, :, :]), axis=1)
        eta_pos_y = np.concatenate((eta_xyz[:, 1:, :, :], eta_xyz[:, -1:, :, :]), axis=1)
        eta_neg_z = np.concatenate((eta_xyz[:1, :, :, :], eta_xyz[:-1, :, :, :]), axis=0)
        eta_pos_z = np.concatenate((eta_xyz[1:, :, :, :], eta_xyz[-1:, :, :, :]), axis=0)

        if pf_args['anisotropy'] > 0.:
            # TODO: Make the code concise
            # https://github.com/google/jax-cfd/blob/8eff9c47bdc7fb19b6453db94ca65f6be64d91f6/jax_cfd/base/finite_differences.py#L74
            centroids = polycrystal.centroids
            centroids_xyz = np.reshape(centroids, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx'], pf_args['dim']))
            edges_x = centroids_xyz[:, :, 1:, :] - centroids_xyz[:, :, :-1, :]
            edges_y = centroids_xyz[:, 1:, :, :] - centroids_xyz[:, :-1, :, :] 
            edges_z = centroids_xyz[1:, :, :, :] - centroids_xyz[:-1, :, :, :]
            aniso_x = update_anisotropy_helper(edges_x)
            aniso_y = update_anisotropy_helper(edges_y)
            aniso_z = update_anisotropy_helper(edges_z)

            aniso_neg_x = np.pad(aniso_x, ((0, 0), (0, 0), (1, 0), (0, 0)))
            aniso_pos_x = np.pad(aniso_x, ((0, 0), (0, 0), (0, 1), (0, 0)))
            aniso_neg_y = np.pad(aniso_y, ((0, 0), (1, 0), (0, 0), (0, 0)))
            aniso_pos_y = np.pad(aniso_y, ((0, 0), (0, 1), (0, 0), (0, 0)))
            aniso_neg_z = np.pad(aniso_z, ((1, 0), (0, 0), (0, 0), (0, 0)))
            aniso_pos_z = np.pad(aniso_z, ((0, 1), (0, 0), (0, 0), (0, 0)))
            print("End of compute_anisotropy...")
        else:
            aniso_neg_x, aniso_pos_x, aniso_neg_y, aniso_pos_y, aniso_neg_z, aniso_pos_z = 1., 1., 1., 1., 1., 1

        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -(np.stack((((eta_pos_x - eta_xyz)*aniso_neg_x + (eta_neg_x - eta_xyz)*aniso_pos_x)/h_x**2, 
                                  ((eta_pos_y - eta_xyz)*aniso_neg_y + (eta_neg_y - eta_xyz)*aniso_pos_y)/h_y**2, 
                                  ((eta_pos_z - eta_xyz)*aniso_neg_z + (eta_neg_z - eta_xyz)*aniso_pos_z)/h_z**2), axis=-1) *
                                   pf_args['kappa_g'] * pf_args['ad_hoc'])

        assert laplace_xyz.shape == (pf_args['Nz'], pf_args['Ny'], pf_args['Nx'], pf_args['num_oris'], pf_args['dim'])
        laplace = np.sum(laplace_xyz.reshape(-1, pf_args['num_oris'], pf_args['dim']), axis=-1)
        assert local_energy_grad.shape == laplace.shape
        Lg = pf_args['L0'] * np.exp(-pf_args['Qg'] / (T*pf_args['gas_const']))
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


def get_force_eta_fn(pf_args):
    def force_eta_zero_in_liquid(state, T):
        '''
        In liquid zone, set all eta to be zero.
        '''
        eta, t = state
        shift_val = 1.
        liquid = T > pf_args['T_melt'] + shift_val
        eta = np.where(liquid, 0., eta)
        return (eta, state[1])
    return jax.jit(force_eta_zero_in_liquid)


class PFSolver:
    def __init__(self, pf_args, polycrystal):
        self.pf_args = pf_args
        self.polycrystal = polycrystal
        self.state_rhs = phase_field(self.polycrystal, self.pf_args)
        self.force_eta_fn = get_force_eta_fn(self.pf_args)

    def stepper(self, state_pre, t_crt, ode_params):
        T, = ode_params
        state, y = explicit_euler(state_pre, t_crt, self.state_rhs, ode_params) 
        state = self.force_eta_fn(state, T)
        return state, state[0]
        
    def ini_cond(self):
        '''
        Prescribe the initial conditions for eta.
        '''
        num_nodes = len(self.polycrystal.centroids)
        eta = np.zeros((num_nodes, self.pf_args['num_oris']))
        eta = eta.at[np.arange(num_nodes), self.polycrystal.cell_ori_inds].set(1)
        y0 = eta
        return y0

    def clean_sols(self):
        '''
        Clean the data folder.
        '''
        vtk_folder = os.path.join(self.pf_args['data_dir'], f"vtk/pf/sols")
        os.makedirs(vtk_folder, exist_ok=True)
        files_vtk = glob.glob(vtk_folder + f"/*")
        for f in files_vtk:
            os.remove(f)

    def inspect_sol(self, pf_sol, pf_sol0, T, ts, step):
        '''
        While running simulations, print out some useful information.
        '''
        # print(np.hstack((T[:100, :], pf_sol[:100, :10])))
        print(f"step {step} of {len(ts[1:])}, unix timestamp = {time.time()}")
        eta0 = np.argmax(pf_sol0, axis=1)
        eta = np.argmax(pf_sol, axis=1)
        change_eta = np.where(eta0 == eta, 0, 1)
        change_T = np.where(T >= self.pf_args['T_melt'], 1, 0)
        print(f"percent of change of orientations = {np.sum(change_eta)/len(change_eta)*100}%")
        print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
        print(f"max T = {np.max(T)}")
 
        if not np.all(np.isfinite(pf_sol)):          
            raise ValueError(f"Found np.inf or np.nan in pf_sol - stop the program")

    def write_sols(self, pf_sol, T, step):
        print(f"Write sols to file...")
        step = step // self.pf_args['write_sol_interval']

        liquid = T.reshape(-1) > self.pf_args['T_melt']
        eta = pf_sol
        eta_max = onp.max(eta, axis=1)
        cell_ori_inds = onp.argmax(eta, axis=1)
        ipf_x = onp.take(self.polycrystal.unique_oris_rgb[0], cell_ori_inds, axis=0)
        ipf_y = onp.take(self.polycrystal.unique_oris_rgb[1], cell_ori_inds, axis=0)
        ipf_z = onp.take(self.polycrystal.unique_oris_rgb[2], cell_ori_inds, axis=0)

        # Set liquid region to be black color
        ipf_x[liquid] = 0.
        ipf_y[liquid] = 0.
        ipf_z[liquid] = 0.

        self.polycrystal.mesh.cell_data['T'] = [onp.array(T, dtype=onp.float32)]
        self.polycrystal.mesh.cell_data['ipf_x'] = [ipf_x]
        self.polycrystal.mesh.cell_data['ipf_y'] = [ipf_y]
        self.polycrystal.mesh.cell_data['ipf_z'] = [ipf_z]
        cell_ori_inds = onp.array(cell_ori_inds, dtype=onp.int32)
        self.polycrystal.mesh.cell_data['ori_inds'] = [cell_ori_inds]

        # TODO: file save manager
        # onp.save(os.path.join(self.pf_args['data_dir'], f"numpy/pf/sols/T_{step:03d}.npy"), T)
        # onp.save(os.path.join(self.pf_args['data_dir'], f"numpy/pf/sols/cell_ori_inds_{step:03d}.npy"), cell_ori_inds)

        self.polycrystal.mesh.write(os.path.join(self.pf_args['data_dir'], f"vtk/pf/sols/u{step:03d}.vtu"))
