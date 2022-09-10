import numpy as onp
import jax
import jax.numpy as np
from src.fem.jax_fem import Mesh, Laplace
from src.fem.apps.multi_scale.arguments import args
from src.fem.apps.multi_scale.trainer import get_nn_batch_forward
from src.fem.apps.multi_scale.utils import tensor_to_flat


class HyperElasticity(Laplace):
    """Three modes: rve, dns, nn
    """
    def __init__(self, name, mesh, mode=None, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None, periodic_bc_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        self.mode = mode
        if self.mode == 'rve':
            self.H_bar = None
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli_rve()
            self.periodic_bc_info = periodic_bc_info
            self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_bc_info
            self.stress_strain_fns = self.stress_strain_fns_rve_dns
        elif self.mode == 'dns':
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli_dns()
            self.stress_strain_fns = self.stress_strain_fns_rve_dns
        elif self.mode == 'nn':
            hyperparam = 'default'
            self.nn_batch_forward = get_nn_batch_forward(hyperparam)
            self.stress_strain_fns = self.stress_strain_fns_nn
        else:
            raise NotImplementedError(f"mode = {self.mode} is not defined.")

    def compute_physics(self, sol, u_grads):
        if self.mode == 'rve':
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim) + self.H_bar[None, :, :]
        else:
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
        vmap_stress, _ = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
        return sigmas

    def stress_strain_fns_rve_dns(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P

        vmap_stress = lambda x: jax.vmap(first_PK_stress)(x, self.E, self.nu)
        vmap_energy = lambda x: jax.vmap(psi)(x, self.E, self.nu)
        return vmap_stress, vmap_energy

    def stress_strain_fns_nn(self):
        def psi(F):
            C = F.T @ F
            C_flat = tensor_to_flat(C)
            energy = self.nn_batch_forward(C_flat[None, :])[0]
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        vmap_stress = jax.vmap(first_PK_stress)
        vmap_energy = jax.vmap(psi)
        return vmap_stress, vmap_energy

    def compute_energy(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        if self.mode == 'rve':
            F_reshape = u_grads.reshape(-1, self.vec, self.dim) + np.eye(self.dim)[None, :, :] + self.H_bar[None, :, :]
        else:
            F_reshape = u_grads.reshape(-1, self.vec, self.dim) + np.eye(self.dim)[None, :, :]
        _, vmap_energy  = self.stress_strain_fns()
        psi = vmap_energy(F_reshape).reshape(u_grads.shape[:2]) # (num_cells, num_quads)
        energy = np.sum(psi * self.JxW)
        return energy

    def fluc_to_disp(self, sol_fluc):
        sol_disp = (self.H_bar @ self.points.T).T + sol_fluc
        return sol_disp

    def compute_moduli_rve(self):
        center = np.array([args.L/2., args.L/2., args.L/2.])
        def E_map(point):
            E = np.where(np.max(np.absolute(point - center)) < args.L*args.ratio, args.E_in, args.E_out) # 1e3, 1e2
            return E
        def nu_map(point):
            nu = np.where(np.max(np.absolute(point - center)) < args.L*args.ratio, args.nu_in, args.nu_out) # 0.3, 0.4
            return nu
        E = jax.vmap(jax.vmap(E_map))(self.physical_quad_points).reshape(-1)
        nu = jax.vmap(jax.vmap(nu_map))(self.physical_quad_points).reshape(-1)
        return E, nu

    def compute_moduli_dns(self):
        center = np.array([args.L/2., args.L/2., args.L/2.])
        def E_map(point):
            E = np.where(np.max(np.absolute(point - center)) < args.L*args.ratio, args.E_in, args.E_out) # 1e3, 1e2
            return E

        def nu_map(point):
            nu = np.where(np.max(np.absolute(point - center)) < args.L*args.ratio, args.nu_in, args.nu_out) # 0.3, 0.4
            return nu

        E = jax.vmap(jax.vmap(E_map))(self.physical_quad_points).reshape(-1)
        nu = jax.vmap(jax.vmap(nu_map))(self.physical_quad_points).reshape(-1)

        return E, nu
